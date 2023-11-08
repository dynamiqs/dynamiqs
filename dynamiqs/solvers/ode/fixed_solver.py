from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import add_tuples, none_to_zeros_like, tqdm


def _assert_multiple_of_dt(dt: float, times: Tensor, name: str):
    # assert that `times` values are multiples of `dt`
    is_multiple = torch.isclose(torch.round(times / dt), times / dt)
    if not torch.all(is_multiple):
        idx_diff = torch.where(~is_multiple)[0][0].item()
        raise ValueError(
            f'For fixed time step solvers, every value of `{name}` must be a multiple'
            f' of the time step `dt`, but `dt={dt:.3e}` and'
            f' `{name}[{idx_diff}]={times[idx_diff].item():.3e}`.'
        )


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
        # assert that `tsave` and `tmeas` values are multiples of `dt`
        _assert_multiple_of_dt(self.dt, self.tsave, 'tsave')
        _assert_multiple_of_dt(self.dt, self.tmeas, 'tmeas')

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
        AdjointFixedAutograd.apply(self, self.y0, *self.options.params)

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
                    a, self.options.params, y, allow_unused=True, retain_graph=True
                )
                dg = none_to_zeros_like(dg, self.options.params)
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
        *params: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver
        ctx.tstop = solver.tstop

        # integrate the ODE forward without storing the graph of operations
        solver.run_nograd()

        # save results and model params
        ctx.save_for_backward(solver.ysave)

        # returning `ysave` is required for custom backward functions
        return solver.ysave, solver.exp_save

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
                torch.zeros_like(_p).to(solver.y_bwd) for _p in solver.options.params
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
            for (_g, _p) in zip(solver.g_bwd, solver.options.params)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, solver.a_bwd, *solver.g_bwd
