from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .ode_forward_qsolver import ODEForwardQSolver
from .progress_bar import tqdm
from .solver_options import ODEAdaptiveStep, ODEFixedStep
from .solver_utils import add_tuples, none_to_zeros_like


class ODEAdjointQSolver(ODEForwardQSolver):
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
    qsolver: ODEAdjointQSolver,
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
    qsolver: ODEAdjointQSolver,
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
        qsolver: ODEAdjointQSolver,
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
