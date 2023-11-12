from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable
from tqdm.std import TqdmWarning

from ..solver import AdjointSolver
from ..utils.utils import tqdm
from .ode_solver import ODESolver


class AdjointODESolver(ODESolver, AdjointSolver):
    """Integrate an augmented ODE of the form $(1) dy / dt = fy(y, t)$ and
    $(2) da / dt = fa(a, t)$ in backward time with initial condition $y(t_0)$ using an
    ODE integrator."""

    def run_adjoint(self):
        AdjointAutograd.apply(self, self.y0, *self.options.params)

    def init_augmented(self, t0: float, y0: Tensor, a0: Tensor) -> tuple:
        return t0, y0, a0

    @abstractmethod
    def integrate_augmented(
        self,
        t0: float,
        t1: float,
        y: Tensor,
        a: Tensor,
        g: tuple[Tensor, ...],
        *args: Any,
    ) -> tuple:
        """Integrates the augmented ODE forward from time `t0` to `t1` (with
        `t0` < `t1` < 0) starting from initial state `(y, a)`."""
        pass


def new_leaf_tensor(x: Tensor) -> Tensor:
    """Return a new leaf tensor sharing the same data as `x`."""
    # create a new tensor `y` sharing the same data as `x` but detached from the graph
    y = x.detach()

    # start tracking operations on the new tensor `y` (i.e. make it a leaf tensor)
    y.requires_grad_(True)

    return y


class AdjointAutograd(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointSolver,
        y0: Tensor,
        *params: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # integrate the ODE forward without storing the computation graph
        solver.run_nograd()

        # save `solver` into context for backward pass
        ctx.solver = solver
        # save `solver.ysave` into context for backward pass
        ctx.save_for_backward(solver.ysave)

        return solver.ysave, solver.exp_save

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, *grads: Tensor) -> tuple[None, Tensor, Tensor]:
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
        ysave = ctx.saved_tensors[0]

        # unpack gradients
        grad_ysave, grad_exp_save = grads

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            if solver.options.save_states:
                y0 = ysave[..., -1, :, :]
                a0 = grad_ysave[..., -1, :, :]
            else:
                y0 = ysave[..., :, :]
                a0 = grad_ysave[..., :, :]
            if len(solver.exp_ops) > 0:
                a0 += (grad_exp_save[..., :, -1, None, None] * solver.exp_ops.mH).sum(
                    dim=-3
                )

            g = tuple(torch.zeros_like(p).to(y0) for p in solver.options.params)

            # initialize time: time is negative-valued and sorted ascendingly during
            # backward integration
            tstop_bwd = np.flip(-solver.tstop, axis=0)
            saved_ini = tstop_bwd[-1] == solver.t0
            if not saved_ini:
                tstop_bwd = np.append(tstop_bwd, 0)
            t0 = tstop_bwd[0]

            # initialize progress bar
            solver.pbar = tqdm(total=-t0, disable=not solver.options.verbose)

            # initialize the ODE routine
            t, y, a, *args = solver.init_augmented(t0, y0, a0)

            # integrate the augmented equation backward between every saved state
            for i, tnext in enumerate(tstop_bwd[1:]):
                y, a, g, *args = solver.integrate_augmented(t, tnext, y, a, g, *args)

                if solver.options.save_states and (i < len(tstop_bwd) - 2 or saved_ini):
                    # replace y with its checkpointed version
                    y = ysave[..., -i - 2, :, :]
                    # update adjoint wrt this time point by adding dL / dy(t)
                    a += grad_ysave[..., -i - 2, :, :]

                # update adjoint wrt this time point by adding dL / de(t)
                if len(solver.exp_ops) > 0 and (i < len(tstop_bwd) - 2 or saved_ini):
                    a += (
                        grad_exp_save[..., :, -i - 2, None, None] * solver.exp_ops.mH
                    ).sum(dim=-3)

                # iterate time
                t = tnext

        # close progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            solver.pbar.close()

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(g, solver.options.params)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, *g
