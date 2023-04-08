from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .adaptive import DormandPrince45
from .progress_bar import tqdm
from .solver_options import AdaptiveStep, Dopri45, FixedStep
from .solver_utils import add_tuples, bexpect, none_to_zeros_like


class ForwardQSolver(ABC):
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


class AdjointQSolver(ForwardQSolver):
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
            tuple of two tensors of shape `(..., m, n)`.
        """
        pass


def odeint(
    qsolver: ForwardQSolver,
    y0: Tensor,
    t_save: Tensor,
    *,
    save_states: bool,
    exp_ops: Tensor,
    gradient_alg: Literal['autograd', 'adjoint'] | None,
    parameters: tuple[nn.Parameter, ...] | None,
) -> tuple[Tensor, Tensor]:
    """Integrate a quantum ODE starting from an initial state.

    Args:
        qsolver:
        y0: Initial quantum state, of shape `(..., m, n)`.
        t_save: Times for which results are saved. The ODE is solved from time `t=0.0`
            to `t=t_save[-1]`.
        save_states:
        exp_ops:
        gradient_alg:
        parameters:

    Returns:
        tuple `(y_save, exp_save)` with
            - `y_save` a tensor of shape `(..., len(t_save), m, n)`
            - `exp_save` a tensor of shape `(..., len(exp_ops), len(t_save))`
    """
    # check arguments
    check_t_save(t_save)

    # raise warning if parameters are defined but not used
    if parameters is not None and gradient_alg != 'adjoint':
        warnings.warn('Parameters were supplied in `odeint` but not used.')

    # dispatch to appropriate odeint subroutine
    args = (qsolver, y0, t_save, save_states, exp_ops)
    if gradient_alg is None:
        return _odeint_inplace(*args)
    elif gradient_alg == 'autograd':
        return _odeint_main(*args)
    elif gradient_alg == 'adjoint':
        return _odeint_adjoint(*args, parameters)
    else:
        raise ValueError(
            f'Automatic differentiation algorithm {gradient_alg} is not defined.'
        )


def _odeint_main(
    qsolver: ForwardQSolver,
    y0: Tensor,
    t_save: Tensor,
    save_states: bool,
    exp_ops: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dispatch the ODE integration to fixed or adaptive time step subroutines."""
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint(qsolver, y0, t_save, save_states, exp_ops)
    elif isinstance(qsolver.options, AdaptiveStep):
        return _adaptive_odeint(qsolver, y0, t_save, save_states, exp_ops)


# For now we use *args and **kwargs for helper methods that are not implemented to ease
# the potential api changes that could occur later. When a method is implemented the
# methods should take the same arguments as all others.
def _odeint_inplace(*args, **kwargs):
    """Integrate a quantum ODE with an in-place solver.

    Simple solution for now so torch does not store gradients.
    TODO Implement a genuine in-place solver.
    """
    with torch.no_grad():
        return _odeint_main(*args, **kwargs)


def _adaptive_odeint(
    qsolver: ForwardQSolver,
    y0: Tensor,
    t_save: Tensor,
    save_states: bool,
    exp_ops: Tensor,
) -> tuple[Tensor, Tensor]:
    """Integrate a quantum ODE with an adaptive time step solver.

    This function integrates an ODE of the form `dy / dt = f(t, y)` with `y(0) = y0`,
    using a Runge-Kutta adaptive time step solver.

    For details about the integration method, see Chapter II.4 of `Hairer et al.,
    Solving Ordinary Differential Equations I (1993), Springer Series in Computational
    Mathematics`.
    """
    # initialize save tensors
    y_save, exp_save = None, None
    batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]
    if save_states:
        y_save = torch.zeros(*batch_sizes, len(t_save), m, n).to(y0)

    if len(exp_ops) > 0:
        exp_save = torch.zeros(*batch_sizes, len(exp_ops), len(t_save)).to(y0)

    # save initial solution
    save_counter = 0
    if t_save[0] == 0.0:
        save_counter += 1
        if save_states:
            y_save[..., save_counter, :, :] = y0
        if len(exp_ops) > 0:
            exp_save[..., save_counter] = bexpect(exp_ops, y0)

    # initialize the adaptive solver
    args = (
        qsolver.forward,
        qsolver.options.factor,
        qsolver.options.min_factor,
        qsolver.options.max_factor,
        qsolver.options.atol,
        qsolver.options.rtol,
        t_save.dtype,
        y0.dtype,
        y0.device,
    )
    if isinstance(qsolver.options, Dopri45):
        solver = DormandPrince45(*args)

    # initialize the ODE routine
    t0 = 0.0
    f0 = solver.f(t0, y0)
    dt = solver.init_tstep(f0, y0, t0)

    # initialize the progress bar
    pbar = tqdm(total=t_save[-1].item(), disable=not qsolver.options.verbose)

    # run the ODE routine
    t, y, ft = t0, y0, f0
    step_counter, max_steps = 0, qsolver.options.max_steps
    while t < t_save[-1] and step_counter < max_steps:
        # if a time in `t_save` is reached, raise a flag and rescale dt accordingly
        if t + dt >= t_save[save_counter]:
            save_flag = True
            dt_old = dt
            dt = float(t_save[save_counter] - t)

        # perform a single solver step of size dt
        ft_new, y_new, y_err = solver.step(ft, y, t, dt)

        # compute estimated error of this step
        error = solver.get_error(y_err, y, y_new)

        # update results if step is accepted
        if error <= 1:
            t, y, ft = t + dt, y_new, ft_new

            # update the progress bar
            pbar.update(dt)

            # save results if flag is raised
            if save_flag:
                if save_states:
                    y_save[..., save_counter, :, :] = y
                if len(exp_ops) > 0:
                    exp_save[..., save_counter] = bexpect(exp_ops, y)
                save_counter += 1

        # return to the original dt and lower the flag
        if save_flag:
            dt = dt_old
            save_flag = False

        # compute the next dt
        dt = solver.update_tstep(dt, error)
        step_counter += 1

    # save last state if not already done
    if not save_states:
        y_save = y

    if len(exp_ops) == 0:
        exp_save = torch.empty(*batch_sizes, len(exp_ops))

    return y_save, exp_save


def _fixed_odeint(
    qsolver: ForwardQSolver,
    y0: Tensor,
    t_save: Tensor,
    save_states: bool,
    exp_ops: Tensor,
) -> tuple[Tensor, Tensor]:
    """Integrate a quantum ODE with a fixed time step solver.

    Note:
        The solver times are defined using `torch.linspace` which ensures that the
        overall solution is evolved from the user-defined time (up to an error of
        `rtol=1e-5`). However, this may induce a small mismatch between the time step
        inside `qsolver` and the time step inside the iteration loop. A small error can
        thus buildup throughout the ODE integration. TODO Fix this.

    Args:
        qsolver:
        y0: Initial quantum state, of shape `(..., m, n)`.
        t_save:
        save_states:
        exp_ops:

    Returns:
        tuple `(y_save, exp_save)` with
            - `y_save` a tensor of shape `(..., len(t_save), m, n)`
            - `exp_save` a tensor of shape `(..., len(exp_ops), len(t_save))`
    """
    # get time step from qsolver
    dt = qsolver.dt

    # assert that `t_save` values are multiples of `dt`
    if not torch.allclose(torch.round(t_save / dt), t_save / dt):
        raise ValueError(
            'Every value of argument `t_save` must be a multiple of the time step `dt`.'
        )

    # initialize save tensors
    y_save, exp_save = None, None
    batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]
    if save_states:
        y_save = torch.zeros(*batch_sizes, len(t_save), m, n).to(y0)

    if len(exp_ops) > 0:
        exp_save = torch.zeros(*batch_sizes, len(exp_ops), len(t_save)).to(y0)

    # define time values
    num_times = torch.round(t_save[-1] / dt).int() + 1
    times = torch.linspace(0.0, t_save[-1], num_times)

    # run the ode routine
    y = y0
    save_counter = 0
    for t in tqdm(times[:-1], disable=not qsolver.options.verbose):
        # save solution
        if t >= t_save[save_counter]:
            if save_states:
                y_save[..., save_counter, :, :] = y
            if len(exp_ops) > 0:
                exp_save[..., save_counter] = bexpect(exp_ops, y)
            save_counter += 1

        # iterate solution
        y = qsolver.forward(t, y)

    # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
    if save_states:
        y_save[..., save_counter, :, :] = y
    else:
        y_save = y

    if len(exp_ops) > 0:
        exp_save[..., save_counter] = bexpect(exp_ops, y)
    else:
        exp_save = torch.empty(*batch_sizes, len(exp_ops))

    return y_save, exp_save


def _odeint_adjoint(
    qsolver: AdjointQSolver,
    y0: Tensor,
    t_save: Tensor,
    save_states: bool,
    exp_ops: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, Tensor]:
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
    if parameters is None:
        raise TypeError(
            'For adjoint state gradient computation, parameters must be passed to the'
            ' solver.'
        )

    return ODEIntAdjoint.apply(qsolver, y0, t_save, save_states, exp_ops, *parameters)


def _odeint_augmented_main(
    qsolver: AdjointQSolver,
    y0: Tensor,
    a0: Tensor,
    g0: tuple[Tensor, ...],
    t_span: Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, Tensor]:
    """Integrate the augmented ODE backward."""
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint_augmented(qsolver, y0, a0, g0, t_span, parameters)
    elif isinstance(qsolver.options, AdaptiveStep):
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
    dt = qsolver.dt

    # check t_span
    if not (t_span.ndim == 1 and len(t_span) == 2):
        raise ValueError(
            f'`t_span` should be a tensor of shape (2,), but has shape {t_span.shape}.'
        )
    if t_span[1] <= t_span[0]:
        raise ValueError('`t_span` should be sorted in ascending order.')

    T = t_span[1] - t_span[0]
    if not torch.allclose(torch.round(T / dt), T / dt):
        raise ValueError('The total time of evolution should be a multiple of dt.')

    # define time values
    times = torch.linspace(t_span[0], t_span[-1], torch.round(T / dt).int() + 1)

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
        save_states: bool,
        exp_ops: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.qsolver = qsolver
        ctx.t_save = t_save if save_states else t_save[-1]

        # solve the ODE forward without storing the graph of operations
        y_save, exp_save = _odeint_inplace(qsolver, y0, t_save, save_states, exp_ops)

        # save results and model parameters
        ctx.save_for_backward(y_save, *parameters)

        return y_save, exp_save

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
    if t_save.dim() != 1 or len(t_save) == 0:
        raise ValueError('Argument `t_save` must be a non-empty 1D tensor.')
    if not torch.all(torch.diff(t_save) > 0):
        raise ValueError(
            'Argument `t_save` must be sorted in strictly ascending order.'
        )
    if not torch.all(t_save >= 0):
        raise ValueError('Argument `t_save` must contain positive values only.')
