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
                'Every value of `tsave` must be a multiple of the time step `dt` for '
                'fixed time step ODE solvers.'
            )
        # assert that `tmeas` values are multiples of `dt`
        if not torch.allclose(torch.round(self.tmeas / self.dt), self.tmeas / self.dt):
            raise ValueError(
                'Every value of `tmeas` must be a multiple of the time step `dt` for '
                'fixed time step ODE solvers.'
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
