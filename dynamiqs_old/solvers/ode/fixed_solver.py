from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..utils.utils import add_tuples, none_to_zeros_like
from .adjoint_ode_solver import AdjointODESolver, new_leaf_tensor
from .ode_solver import ODESolver


def _assert_multiple_of_dt(dt: float, times: np.array, name: str):
    # assert that `times` values are multiples of `dt`
    is_multiple = np.isclose(np.round(times / dt), times / dt)
    if not np.all(is_multiple):
        idx_diff = np.where(~is_multiple)[0][0]
        raise ValueError(
            f'For fixed time step solvers, every value of `{name}` must be a multiple'
            f' of the time step `dt`, but `dt={dt:.3e}` and'
            f' `{name}[{idx_diff}]={times[idx_diff]:.3e}`.'
        )


class FixedSolver(ODESolver):
    """Fixed step-size ODE integrator."""

    def __init__(self, *args):
        super().__init__(*args)

        self.dt = self.options.dt

        # assert that `tsave` and `tmeas` values are multiples of `dt`
        _assert_multiple_of_dt(self.dt, self.tsave, 'tsave')
        _assert_multiple_of_dt(self.dt, self.tmeas, 'tmeas')

    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        """Returns $y(t+dt)$."""
        pass

    def integrate(self, t0: float, t1: float, y: Tensor, *args: Any) -> tuple:
        # define time values
        ntimes = int(round((t1 - t0) / self.dt)) + 1
        times = np.linspace(t0, t1, ntimes)

        # TODO: The solver times are defined using `torch.linspace` which ensures that
        # the overall solution is evolved from the user-defined time (up to an error of
        # `rtol=1e-5`). However, this may induce a small mismatch between the time step
        # inside `solver` and the time step inside the iteration loop. A small error
        # can thus buildup throughout the ODE integration.

        # run the ODE routine
        for t in times[:-1]:
            y = self.forward(t, y)
            self.pbar.update(self.dt)

        return (y,)


class AdjointFixedSolver(FixedSolver, AdjointODESolver):
    """Fixed step-size ODE integrator."""

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns $y(t-dt)$ and $a(t-dt)$."""
        pass

    def integrate_augmented(
        self,
        t0: float,
        t1: float,
        y: Tensor,
        a: Tensor,
        g: tuple[Tensor, ...],
        *args: Any,
    ) -> tuple:
        # define time values
        num_times = round((t1 - t0) / self.dt) + 1
        times = np.linspace(t0, t1, num_times)

        # run the ODE routine
        for t in times[:-1]:
            # the computation graph attached to `y` and `a` is automatically freed after
            # next line, because there are no more references to the original tensors
            # (the old `y` and `a` go out of scope)
            y, a = new_leaf_tensor(y), new_leaf_tensor(a)

            with torch.enable_grad():
                # compute y(t-dt) and a(t-dt)
                y, a = self.backward_augmented(-t, y, a)

                # compute g(t-dt)
                # note: we set `retain_graph=True` to keep tracking operations on
                # `self.options.params` in the graph
                dg = torch.autograd.grad(
                    a, self.options.params, y, allow_unused=True, retain_graph=True
                )
                dg = none_to_zeros_like(dg, self.options.params)
                g = add_tuples(g, dg)

            # update progress bar
            self.pbar.update(self.dt)

        # save final augmented state to the solver
        return y, a, g
