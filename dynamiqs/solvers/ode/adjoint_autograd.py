from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..solver import AdjointSolver
from ..utils.utils import tqdm


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
        ctx.tstop = solver.tstop

        # integrate the ODE forward without storing the graph of operations
        solver.run_nograd()

        # save results and model parameters
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
