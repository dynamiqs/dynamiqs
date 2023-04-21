from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .adaptive import DormandPrince45
from .progress_bar import tqdm
from .qsolver import QSolver
from .solver_options import Dopri45, ODEAdaptiveStep, ODEFixedStep
from .solver_utils import add_tuples, none_to_zeros_like


class ForwardQSolver(QSolver):
    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        """Iterate the quantum state forward.

        Args:
            t: Time.
            y: Quantum state, of shape `(..., m, n)`.

        Returns:
            Tensor of shape `(..., m, n)`.
        """
        pass

    def run(self):
        """Integrate a quantum ODE starting from an initial state.

        The ODE is solved from time `t=0.0` to `t=t_save[-1]`.
        """
        # check arguments
        check_t_save(self.t_save)

        # dispatch to appropriate odeint subroutine
        if self.gradient_alg is None:
            self._odeint_inplace()
        elif self.gradient_alg == 'autograd':
            self._odeint_main()

    def _odeint_inplace(self):
        """Integrate a quantum ODE with an in-place solver.

        Simple solution for now so torch does not store gradients.
        TODO Implement a genuine in-place solver.
        """
        with torch.no_grad():
            self._odeint_main()

    def _odeint_main(self):
        """Dispatch the ODE integration to fixed or adaptive time step subroutines."""
        if isinstance(self.options, ODEFixedStep):
            self._fixed_odeint()
        elif isinstance(self.options, ODEAdaptiveStep):
            self._adaptive_odeint()

    def _fixed_odeint(self):
        """Integrate a quantum ODE with a fixed time step solver.

        Note:
            The solver times are defined using `torch.linspace` which ensures that the
            overall solution is evolved from the user-defined time (up to an error of
            `rtol=1e-5`). However, this may induce a small mismatch between the time
            step inside `qsolver` and the time step inside the iteration loop. A small
            error can thus buildup throughout the ODE integration. TODO Fix this.
        """
        # get time step
        dt = self.options.dt

        # assert that `t_save` values are multiples of `dt`
        if not torch.allclose(torch.round(self.t_save / dt), self.t_save / dt):
            raise ValueError(
                'For fixed time step solvers, every value of `t_save` must be a'
                ' multiple of the time step `dt`.'
            )

        # define time values
        num_times = torch.round(self.t_save[-1] / dt).int() + 1
        times = torch.linspace(0.0, self.t_save[-1], num_times)

        # run the ode routine
        y = self.y0
        for t in tqdm(times[:-1], disable=not self.options.verbose):
            # save solution
            if t >= self.next_tsave():
                self.save(y)

            # iterate solution
            y = self.forward(t, y)

        # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
        self.save(y)
        self.save_final(y)

    def _adaptive_odeint(self):
        """Integrate a quantum ODE with an adaptive time step solver.

        This function integrates an ODE of the form `dy / dt = f(t, y)` with
        `y(0) = y0`, using a Runge-Kutta adaptive time step solver.

        For details about the integration method, see Chapter II.4 of `Hairer et al.,
        Solving Ordinary Differential Equations I (1993), Springer Series in
        Computational Mathematics`.
        """
        # save initial solution
        self.save(self.y0)

        save_flag = False

        # initialize the adaptive solver
        args = (
            self.forward,
            self.options.factor,
            self.options.min_factor,
            self.options.max_factor,
            self.options.atol,
            self.options.rtol,
        )
        if isinstance(self.options, Dopri45):
            solver = DormandPrince45(*args)

        # initialize the ODE routine
        t0 = 0.0
        f0 = solver.f(t0, self.y0)
        dt = solver.init_tstep(f0, self.y0, t0)

        # initialize the progress bar
        pbar = tqdm(total=self.t_save[-1].item(), disable=not self.options.verbose)

        # run the ODE routine
        t, y, ft = t0, self.y0, f0
        step_counter, max_steps = 0, self.options.max_steps
        next_tsave = self.next_tsave()
        while t < self.t_save[-1] and step_counter < max_steps:
            # if a time in `t_save` is reached, raise a flag and rescale dt accordingly
            if t + dt >= next_tsave:
                save_flag = True
                dt_old = dt
                dt = next_tsave - t

            # perform a single solver step of size dt
            ft_new, y_new, y_err = solver.step(ft, y, t, dt)

            # compute estimated error of this step
            error = solver.get_error(y_err, y, y_new)

            # update results if step is accepted
            if error <= 1:
                t, y, ft = t + dt, y_new, ft_new

                # update the progress bar
                pbar.update(dt.item())

                # save results if flag is raised
                if save_flag:
                    self.save(t, y)

            # return to the original dt, lower the flag and get next save time
            if save_flag:
                dt = dt_old
                save_flag = False
                next_tsave = self.next_tsave()

            # compute the next dt
            dt = solver.update_tstep(dt, error)
            step_counter += 1

        # save last state if not already done
        self.save_final(y)


class AdjointQSolver(ForwardQSolver):
    GRADIENT_ALG = ['autograd', 'adjoint']

    def run(self):
        super().run()

        if self.gradient_alg == 'adjoint':
            self._odeint_adjoint()

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor, parameters: tuple[nn.Parameter, ...]
    ) -> tuple[Tensor, Tensor]:
        """Iterate the augmented quantum state backward.

        Args:
            t: Time.
            y: Quantum state, of shape `(..., m, n)`.
            a: Adjoint quantum state, of shape `(..., m, n)`.
            parameters (tuple of nn.Parameter): Parameters w.r.t. compute the gradients.

        Returns:
            Tuple of two tensors of shape `(..., m, n)`.
        """
        pass

    def _odeint_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass.

        Within this function, the following calls are sequentially made:
        - Forward pass: `odeint` --> `_odeint_adjoint` --> `ODEIntAdjoint.forward` -->
                        `_odeint_inplace` --> `_odeint_main` --> `_fixed_odeint` or
                        `_adaptive_odeint` --> `qsolver.forward`.
        - Backward pass: `odeint` --> `_odeint_adjoint` --> `ODEIntAdjoint.backward` -->
                        `_odeint_augmented_main` --> `_fixed_odeint_augmented` or
                        `_adaptive_odeint_augmented` --> `qsolver.backward_augmented`.
        """
        # check parameters were passed
        if self.parameters is None:
            raise TypeError(
                'For adjoint state gradient computation, parameters must be passed to'
                ' the solver.'
            )

        ODEIntAdjoint.apply(self, self.y0, self.t_save, *self.parameters)


def _odeint_augmented_main(
    qsolver: AdjointQSolver,
    y0: Tensor,
    a0: Tensor,
    g0: tuple[Tensor, ...],
    t_span: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, Tensor]:
    """Integrate the augmented ODE backward."""
    if isinstance(qsolver.options, ODEFixedStep):
        return _fixed_odeint_augmented(qsolver, y0, a0, g0, t_span, parameters)
    elif isinstance(qsolver.options, ODEAdaptiveStep):
        return _adaptive_odeint_augmented(qsolver, y0, a0, g0, t_span, parameters)


def _adaptive_odeint_augmented(*_args, **_kwargs):
    """Integrate the augmented ODE backward using an adaptive time step solver."""
    raise NotImplementedError


def _fixed_odeint_augmented(
    qsolver: AdjointQSolver,
    y0: Tensor,
    a0: Tensor,
    g0: tuple[Tensor, ...],
    t_span: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Integrate the augmented ODE backward using a fixed time step solver."""
    # get time step from qsolver
    dt = qsolver.options.dt

    # check t_span
    if not (t_span.ndim == 1 and len(t_span) == 2):
        raise ValueError(
            f'`t_span` should be a tensor of size (2,), but has size {t_span.shape}.'
        )
    if t_span[1] <= t_span[0]:
        raise ValueError('`t_span` should be sorted in ascending order.')

    T = t_span[1] - t_span[0]
    if not torch.allclose(torch.round(T / dt), T / dt):
        raise ValueError(
            'For fixed time step adjoint solvers, every value of `t_save` must be a '
            'multiple of the time step `dt`.'
        )

    # define time values
    num_times = torch.round(T / dt).int() + 1
    times = torch.linspace(t_span[1], t_span[0], num_times)

    # run the ode routine
    y, a, g = y0, a0, g0
    for t in tqdm(times[:-1], leave=False, disable=not qsolver.options.verbose):
        y, a = y.requires_grad_(True), a.requires_grad_(True)

        with torch.enable_grad():
            # compute y(t-dt) and a(t-dt) with the qsolver
            y, a = qsolver.backward_augmented(t, y, a, parameters)

            # compute g(t-dt)
            dg = torch.autograd.grad(
                a, parameters, y, allow_unused=True, retain_graph=True
            )
            dg = none_to_zeros_like(dg, parameters)
            g = add_tuples(g, dg)

        # free the graph of y and a
        y, a = y.data, a.data

    return y, a, g


class ODEIntAdjoint(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        qsolver: AdjointQSolver,
        y0: Tensor,
        t_save: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ):
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.qsolver = qsolver
        ctx.t_save = t_save if qsolver.options.save_states else t_save[-1]

        # solve the ODE forward without storing the graph of operations
        qsolver._odeint_inplace()

        # save results and model parameters
        ctx.save_for_backward(qsolver.y_save, *parameters)

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_y: Tensor):
        """Backward pass of the ODE integrator.

        An augmented ODE is integrated backwards starting from the final state computed
        during the forward pass. Integration is done in multiple sequential runs
        between every checkpoint of the forward pass, as defined by `t_save`. This
        helps with the stability of the backward integration.

        Throughout this function, `y` is the state, `a = dL/dy` is the adjoint state,
        and `g = dL/dp` is the gradient w.r.t. the parameters, where `L` is the loss
        function and `p` the parameters.
        """
        # unpack context
        qsolver = ctx.qsolver
        t_save = ctx.t_save
        y_save, *parameters = ctx.saved_tensors

        # initialize time list
        t_span = t_save
        if t_span[0] != 0.0:
            t_span = torch.cat((torch.zeros(1), t_span))

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            y = y_save[..., -1, :, :]
            a = grad_y[0][..., -1, :, :]
            g = tuple(torch.zeros_like(_p).to(y) for _p in parameters)

            # solve the augmented equation backward between every checkpoint
            for i in tqdm(
                range(len(t_span) - 1, 0, -1), disable=not qsolver.options.verbose
            ):
                # initialize time between both checkpoints
                t_span_segment = t_span[i - 1 : i + 1]

                # run odeint on augmented state
                y, a, g = _odeint_augmented_main(
                    qsolver, y, a, g, t_span_segment, parameters=parameters
                )

                # replace y with its checkpointed version
                y = y_save[..., i - 1, :, :]

                # update adjoint wrt this time point by adding dL / dy(t)
                a += grad_y[0][..., i - 1, :, :]

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g for (_g, _p) in zip(g, parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, None, None, None, *g


def check_t_save(t_save: Tensor):
    """Check that `t_save` is valid (it must be a non-empty 1D tensor sorted in
    strictly ascending order and containing only positive values)."""
    if t_save.ndim != 1 or len(t_save) == 0:
        raise ValueError('Argument `t_save` must be a non-empty 1D tensor.')
    if not torch.all(torch.diff(t_save) > 0):
        raise ValueError(
            'Argument `t_save` must be sorted in strictly ascending order.'
        )
    if not torch.all(t_save >= 0):
        raise ValueError('Argument `t_save` must contain positive values only.')
