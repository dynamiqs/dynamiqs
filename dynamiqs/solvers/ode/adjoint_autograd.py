from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx
from tqdm.std import TqdmWarning

from ..solver import AdjointSolver
from ..utils.utils import tqdm

class AdjointAdaptiveAutograd(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointSolver,
        y0: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver
        ctx.tsave = solver.tsave

        # integrate the ODE forward without storing the graph of operations
        with torch.no_grad():
            solver.run_autograd()

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
        tsave = ctx.tsave
        ysave = ctx.saved_tensors[0]

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            if solver.options.save_states:
                y = ysave[..., -1, :, :]
                a = grad_y[0][..., -1, :, :]
            else:
                y = ysave[..., :, :]
                a = grad_y[0][..., :, :]
            if len(solver.exp_ops) > 0:
                a += (grad_y[1][..., :, -1, None, None] * solver.exp_ops.mH).sum(dim=-3)

            g = tuple(torch.zeros_like(p).to(y) for p in solver.options.parameters)

            # initialize time
            tstop = solver.tstop_backward
            T = -tsave[-1].item()

            # initialize the progress bar
            nobar = not solver.options.verbose
            solver.pbar = tqdm(total=-T, disable=nobar)

            # initialize the ODE routine
            f0, l0 = solver.odefun_augmented(T, y, a)
            dt_y = solver.init_tstep(f0, y, T, solver.odefun_backward)
            dt_a = solver.init_tstep(l0, a, T, solver.odefun_adjoint)
            dt = min(dt_y, dt_a)
            error = 1.0

            # integrate the augmented equation backward between every saved state
            t, ft, lt = T, f0, l0
            for i, ts in enumerate(tstop):
                y, a, g, ft, lt, dt, error = solver.integrate_augmented(
                    t, ts, y, a, g, ft, lt, dt, error
                )

                if solver.options.save_states:
                    # replace y with its checkpointed version
                    y = ysave[..., -i - 2, :, :]
                    # update adjoint wrt this time point by adding dL / dy(t)
                    a += grad_y[0][..., -i - 2, :, :]

                # update adjoint wrt this time point by adding dL / de(t)
                if len(solver.exp_ops) > 0:
                    a += (
                        grad_y[1][..., :, -i - 2, None, None] * solver.exp_ops.mH
                    ).sum(dim=-3)

                # iterate time
                t = ts

        # close progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            solver.pbar.close()

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(g, solver.options.parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, *g


class AdjointFixedAutograd(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointSolver,
        y0: Tensor,
        *parameters: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver
        ctx.tsave = solver.tsave

        # integrate the ODE forward without storing the graph of operations
        with torch.no_grad():
            solver.run_autograd()

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
        tsave = ctx.tsave
        ysave = ctx.saved_tensors[0]

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            if solver.options.save_states:
                y = ysave[..., -1, :, :]
                a = grad_y[0][..., -1, :, :]
            else:
                y = ysave[..., :, :]
                a = grad_y[0][..., :, :]
            if len(solver.exp_ops) > 0:
                a += (grad_y[1][..., :, -1, None, None] * solver.exp_ops.mH).sum(dim=-3)

            g = tuple(torch.zeros_like(p).to(y) for p in solver.options.parameters)

            # initialize time
            t = -tsave[-1].item()
            tstop = solver.tstop_backward

            # integrate the augmented equation backward between every saved state
            nobar = not solver.options.verbose
            for i, ts in enumerate(tqdm(tstop, disable=nobar)):
                y, a, g = solver.integrate_augmented(t, ts, y, a, g)

                if solver.options.save_states:
                    # replace y with its checkpointed version
                    y = ysave[..., -i - 2, :, :]
                    # update adjoint wrt this time point by adding dL / dy(t)
                    a += grad_y[0][..., -i - 2, :, :]

                # update adjoint wrt this time point by adding dL / de(t)
                if len(solver.exp_ops) > 0:
                    a += (
                        grad_y[1][..., :, -i - 2, None, None] * solver.exp_ops.mH
                    ).sum(dim=-3)

                # iterate time
                t = ts

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(g, solver.options.parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, *g
