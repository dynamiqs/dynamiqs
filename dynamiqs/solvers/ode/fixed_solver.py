from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import add_tuples, none_to_zeros_like, tqdm


class FixedSolver(AutogradSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.dt = self.options.dt

    def run_autograd(self):
        """Integrate a quantum ODE with a fixed time step custom integrator.

        Notes:
            The solver times are defined using `torch.linspace` which ensures that the
            overall solution is evolved from the user-defined time (up to an error of
            `rtol=1e-5`). However, this may induce a small mismatch between the time
            step inside `solver` and the time step inside the iteration loop. A small
            error can thus buildup throughout the ODE integration. TODO Fix this.
        """
        # assert that `tstop` values are multiples of `dt`. If not raise an error.
        if not torch.allclose(torch.round(self.tstop / self.dt), self.tstop / self.dt):
            if hasattr(self, 'tmeas'):
                tmeas_indices = torch.nonzero(
                    torch.round(self.tmeas / self.dt) != self.tmeas / self.dt
                )
                if len(tmeas_indices) > 0:
                    idx = tmeas_indices[0]
                    raise ValueError(
                        'For fixed time step solvers, every value of `tmeas` must be a'
                        f' multiple of the time step `dt`, but `dt = {self.dt:.5e}` and'
                        f' `tmeas[{idx.item()}] = {self.tmeas[idx].item():.5e}`.'
                    )

            tsave_indices = torch.nonzero(
                torch.round(self.tsave / self.dt) != self.tsave / self.dt
            )
            idx = tsave_indices[0]
            raise ValueError(
                'For fixed time step solvers, every value of `tsave` must be a'
                f' multiple of the time step `dt`, but `dt = {self.dt:.5e}` and'
                f' `tsave[{idx.item()}] = {self.tsave[idx].item():.5e}`.'
            )

        # define time values
        num_times = torch.round(self.tstop[-1] / self.dt).int() + 1
        times = torch.linspace(0.0, self.tstop[-1], num_times)

        # run the ode routine
        y = self.y0
        for t in tqdm(times[:-1].cpu().numpy(), disable=not self.options.verbose):
            # save solution
            if t >= self.next_tstop():
                self.save(y)

            # iterate solution
            y = self.forward(t, y)

        # save final time step (`t` goes `0.0` to `tstop[-1]` excluded)
        self.save(y)

    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        pass


class AdjointFixedSolver(FixedSolver, AdjointSolver):
    def run_adjoint(self):
        AdjointFixedAutograd.apply(self, self.y0, *self.options.parameters)

    def run_augmented(self):
        """Integrate the augmented ODE backward using a fixed time step integrator."""
        # check tstop_bwd
        if not self.tstop_bwd.ndim == 1:
            raise ValueError(
                'Attribute `tstop_bwd` must be a must be a 1D tensor, but is a'
                f' {self.tstop_bwd.ndim}D tensor.'
            )
        if not len(self.tstop_bwd) == 2:
            raise ValueError(
                'Attribute `tstop_bwd` must have length 2, but has length'
                f' {len(self.tstop_bwd)}.'
            )
        if self.tstop_bwd[1] <= self.tstop_bwd[0]:
            raise ValueError(
                'Attribute `tstop_bwd` must be sorted in strictly ascending order.'
            )

        T = self.tstop_bwd[1] - self.tstop_bwd[0]
        if not torch.allclose(torch.round(T / self.dt), T / self.dt):
            raise ValueError(
                'Every value of `tstop_bwd` must be a multiple of the time step `dt`'
                ' for fixed time step ODE solvers.'
            )

        # define time values
        num_times = torch.round(T / self.dt).int() + 1
        times = torch.linspace(self.tstop_bwd[1], self.tstop_bwd[0], num_times)

        # run the ode routine
        y, a, g = self.y_bwd, self.a_bwd, self.g_bwd
        for t in tqdm(times[:-1], leave=False, disable=not self.options.verbose):
            y, a = y.requires_grad_(True), a.requires_grad_(True)

            with torch.enable_grad():
                # compute y(t-dt) and a(t-dt) with the solver
                y, a = self.backward_augmented(t, y, a)

                # compute g(t-dt)
                dg = torch.autograd.grad(
                    a, self.options.parameters, y, allow_unused=True, retain_graph=True
                )
                dg = none_to_zeros_like(dg, self.options.parameters)
                g = add_tuples(g, dg)

            # free the graph of y and a
            y, a = y.data, a.data

        # save final augmented state to the solver
        self.y_bwd = y
        self.a_bwd = a
        self.g_bwd = g

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass


class AdjointFixedAutograd(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointFixedSolver,
        y0: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver
        ctx.tstop = solver.tstop

        # integrate the ODE forward without storing the graph of operations
        solver.run_nograd()

        # save results and model parameters
        ctx.save_for_backward(solver.result.ysave)

        # returning `ysave` is required for custom backward functions
        return solver.result.ysave, solver.result.exp_save

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_y: Tensor) -> tuple[None, Tensor, Tensor]:
        """Backward pass of the ODE integrator.

        An augmented ODE is integrated backwards starting from the final state computed
        during the forward pass. Integration is done in multiple sequential runs
        between every checkpoint of the forward pass, as defined by `tstop`. This
        helps with the stability of the backward integration.

        Throughout this function, `y` is the state, `a = dL/dy` is the adjoint state,
        and `g = dL/dp` is the gradient w.r.t. the parameters, where `L` is the loss
        function and `p` the parameters.
        """
        # unpack context
        solver = ctx.solver
        tstop = ctx.tstop
        ysave = ctx.saved_tensors[0]

        # initialize time list
        tstop = tstop
        if tstop[0] != 0.0:
            tstop = torch.cat((torch.zeros(1), tstop))

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            if solver.options.save_states:
                solver.y_bwd = ysave[..., -1, :, :]
                solver.a_bwd = grad_y[0][..., -1, :, :]
            else:
                solver.y_bwd = ysave[..., :, :]
                solver.a_bwd = grad_y[0][..., :, :]
            if len(solver.exp_ops) > 0:
                solver.a_bwd += (
                    grad_y[1][..., :, -1, None, None] * solver.exp_ops.mH
                ).sum(dim=-3)

            solver.g_bwd = tuple(
                torch.zeros_like(_p).to(solver.y_bwd)
                for _p in solver.options.parameters
            )

            # solve the augmented equation backward between every checkpoint
            for i in tqdm(
                range(len(tstop) - 1, 0, -1), disable=not solver.options.verbose
            ):
                # initialize time between both checkpoints
                solver.tstop_bwd = tstop[i - 1 : i + 1]

                # run odeint on augmented state
                solver.run_augmented()

                if solver.options.save_states:
                    # replace y with its checkpointed version
                    solver.y_bwd = ysave[..., i - 1, :, :]
                    # update adjoint wrt this time point by adding dL / dy(t)
                    solver.a_bwd += grad_y[0][..., i - 1, :, :]

                # update adjoint wrt this time point by adding dL / de(t)
                if len(solver.exp_ops) > 0:
                    solver.a_bwd += (
                        grad_y[1][..., :, i - 1, None, None] * solver.exp_ops.mH
                    ).sum(dim=-3)

        # convert gradients of real-valued parameters to real-valued gradients
        solver.g_bwd = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(solver.g_bwd, solver.options.parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, solver.a_bwd, *solver.g_bwd
