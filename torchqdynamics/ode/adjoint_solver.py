from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..options import ODEAdaptiveStep, ODEFixedStep
from ..utils.progress_bar import tqdm
from ..utils.solver_utils import add_tuples, none_to_zeros_like
from .forward_solver import ForwardSolver


class AdjointSolver(ForwardSolver):
    GRADIENT_ALG = ['autograd', 'adjoint']

    def run(self):
        if self.gradient_alg in [None, 'autograd']:
            super().run()
        elif self.gradient_alg == 'adjoint':
            self._odeint_adjoint()

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor
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
        - Forward pass: `run` --> `_odeint_adjoint` --> `ODEIntAdjoint.forward` -->
                        `_odeint_inplace` --> `_odeint_main` --> `_fixed_odeint` or
                        `_adaptive_odeint` --> `solver.forward`.
        - Backward pass: `run` --> `_odeint_adjoint` --> `ODEIntAdjoint.backward` -->
                        `_odeint_augmented_main` --> `_fixed_odeint_augmented` or
                        `_adaptive_odeint_augmented` --> `solver.backward_augmented`.
        """
        # check parameters were passed
        if self.parameters is None:
            raise TypeError(
                'For adjoint state gradient computation, parameters must be passed to'
                ' the solver.'
            )

        ODEIntAdjoint.apply(self, self.y0, *self.parameters)

    def _odeint_augmented_main(self):
        """Integrate the augmented ODE backward."""
        if isinstance(self.options, ODEFixedStep):
            self._fixed_odeint_augmented()
        elif isinstance(self.options, ODEAdaptiveStep):
            self._adaptive_odeint_augmented()

    def _fixed_odeint_augmented(self):
        """Integrate the augmented ODE backward using a fixed time step integrator."""
        # get time step from solver
        dt = self.options.dt

        # check t_save_bwd
        if not (self.t_save_bwd.ndim == 1 and len(self.t_save_bwd) == 2):
            raise ValueError(
                '`t_save_bwd` should be a tensor of size (2,), but has size'
                f' {self.t_save_bwd.shape}.'
            )
        if self.t_save_bwd[1] <= self.t_save_bwd[0]:
            raise ValueError('`t_save_bwd` should be sorted in ascending order.')

        T = self.t_save_bwd[1] - self.t_save_bwd[0]
        if not torch.allclose(torch.round(T / dt), T / dt):
            raise ValueError(
                'For fixed time step adjoint solvers, every value of `t_save_bwd` '
                'must be a multiple of the time step `dt`.'
            )

        # define time values
        num_times = torch.round(T / dt).int() + 1
        times = torch.linspace(self.t_save_bwd[1], self.t_save_bwd[0], num_times)

        # run the ode routine
        y, a, g = self.y_bwd, self.a_bwd, self.g_bwd
        for t in tqdm(times[:-1], leave=False, disable=not self.options.verbose):
            y, a = y.requires_grad_(True), a.requires_grad_(True)

            with torch.enable_grad():
                # compute y(t-dt) and a(t-dt) with the solver
                y, a = self.backward_augmented(t, y, a)

                # compute g(t-dt)
                dg = torch.autograd.grad(
                    a, self.parameters, y, allow_unused=True, retain_graph=True
                )
                dg = none_to_zeros_like(dg, self.parameters)
                g = add_tuples(g, dg)

            # free the graph of y and a
            y, a = y.data, a.data

        # save final augmented state to the solver
        self.y_bwd = y
        self.a_bwd = a
        self.g_bwd = g

    def _adaptive_odeint_augmented(self, *_args, **_kwargs):
        """Integrate the augmented ODE backward with an adaptive step integrator."""
        raise NotImplementedError


class ODEIntAdjoint(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointSolver,
        y0: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ) -> Tensor:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver
        ctx.t_save = solver.t_save if solver.options.save_states else solver.t_save[-1]

        # integrate the ODE forward without storing the graph of operations
        solver._odeint_inplace()

        # save results and model parameters
        ctx.save_for_backward(solver.y_save)

        # returning `y_save` is required for custom backward functions
        return solver.y_save

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_y: Tensor) -> tuple[None, Tensor, Tensor]:
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
        solver = ctx.solver
        t_save = ctx.t_save
        y_save = ctx.saved_tensors

        # initialize time list
        t_save = t_save
        if t_save[0] != 0.0:
            t_save = torch.cat((torch.zeros(1), t_save))

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            solver.y_bwd = y_save[..., -1, :, :]
            solver.a_bwd = grad_y[0][..., -1, :, :]
            solver.g_bwd = tuple(
                torch.zeros_like(_p).to(solver.y_bwd) for _p in solver.parameters
            )

            # solve the augmented equation backward between every checkpoint
            for i in tqdm(
                range(len(t_save) - 1, 0, -1), disable=not solver.options.verbose
            ):
                # initialize time between both checkpoints
                solver.t_save_bwd = t_save[i - 1 : i + 1]

                # run odeint on augmented state
                solver._odeint_augmented_main()

                # replace y with its checkpointed version
                solver.y_bwd = y_save[..., i - 1, :, :]

                # update adjoint wrt this time point by adding dL / dy(t)
                solver.a_bwd += grad_y[0][..., i - 1, :, :]

        # convert gradients of real-valued parameters to real-valued gradients
        solver.g_bwd = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(solver.g_bwd, solver.parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, solver.a_bwd, *solver.g_bwd
