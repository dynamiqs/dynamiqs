from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import add_tuples, none_to_zeros_like, tqdm
from .adjoint_autograd import AdjointFixedAutograd


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
        # assert that `tsave` values are multiples of `dt`
        if not torch.allclose(torch.round(self.tsave / self.dt), self.tsave / self.dt):
            raise ValueError(
                'Every value of `tsave` (and `tmeas` for SME solvers) must be a'
                ' multiple of the time step `dt` for fixed time step ODE solvers.'
            )

        # initialize time and state
        t, y = 0.0, self.y0

        # run the ode routine
        nobar = not self.options.verbose
        for ts in tqdm(self.tstop, disable=nobar):
            # integrate the ODE forward
            y = self.integrate(t, ts, y)
            self.save(y)
            t = ts

    def integrate(self, t0: float, t1: float, y: Tensor) -> Tensor:
        """Integrate the ODE forward from time `t0` to time `t1`."""
        # define time values
        num_times = round((t1 - t0) / self.dt) + 1
        times = torch.linspace(t0, t1, num_times)

        # run the ode routine
        nobar = not self.options.verbose
        for t in tqdm(times[:-1].cpu().numpy(), leave=False, disable=nobar):
            y = self.forward(t, y)

        return y

    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        pass


class AdjointFixedSolver(FixedSolver, AdjointSolver):
    def run_adjoint(self):
        AdjointFixedAutograd.apply(self, self.y0, *self.options.parameters)

    def integrate_augmented(
        self, t1: float, t0: float, y: Tensor, a: Tensor, g: tuple[Tensor, ...]
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Integrate the augmented ODE backward from time `t1` to `t0`."""
        # define time values
        num_times = round((t1 - t0) / self.dt) + 1
        times = torch.linspace(t1, t0, num_times)

        # run the ode routine
        nobar = not self.options.verbose
        for t in tqdm(times[:-1].cpu().numpy(), leave=False, disable=nobar):
            y, a = y.requires_grad_(True), a.requires_grad_(True)

            with torch.enable_grad():
                # compute y(t-dt) and a(t-dt)
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
        return y, a, g

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass
