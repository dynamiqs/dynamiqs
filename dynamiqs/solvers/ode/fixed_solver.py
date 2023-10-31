from __future__ import annotations

import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from tqdm.std import TqdmWarning

from ..solver import AdjointSolver, AutogradSolver
from ..utils.utils import add_tuples, none_to_zeros_like, tqdm
from .adjoint_autograd import AdjointAutograd


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

        # initialize the progress bar
        self.pbar = tqdm(total=self.tstop[-1].item(), disable=not self.options.verbose)

        # initialize time and state
        t, y = self.t0, self.y0

        # run the ode routine
        for ts in self.tstop.cpu().numpy():
            y = self.integrate(t, ts, y)
            self.save(y)
            t = ts

        # close progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            self.pbar.close()

    def integrate(self, t0: float, t1: float, y: Tensor) -> Tensor:
        """Integrate the ODE forward from time `t0` to time `t1`."""
        # define time values
        num_times = round((t1 - t0) / self.dt) + 1
        times = torch.linspace(t0, t1, num_times)

        # run the ode routine
        for t in times[:-1].cpu().numpy():
            y = self.forward(t, y)
            self.pbar.update(self.dt)

        return y

    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        pass


class AdjointFixedSolver(FixedSolver, AdjointSolver):
    def run_adjoint(self):
        AdjointAutograd.apply(self, self.y0, *self.options.params)

    def init_augmented(self, t0: float, y: Tensor, a: Tensor) -> tuple:
        return ()

    def integrate_augmented(
        self, t0: float, t1: float, y: Tensor, a: Tensor, g: tuple[Tensor, ...]
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Integrate the augmented ODE from time `t0` to `t1`, with `t0` < `t1` < 0."""
        # define time values
        num_times = round((t1 - t0) / self.dt) + 1
        times = torch.linspace(t0, t1, num_times)

        # run the ode routine
        for t in times[:-1].cpu().numpy():
            y, a = y.requires_grad_(True), a.requires_grad_(True)

            with torch.enable_grad():
                # compute y(t-dt) and a(t-dt)
                y, a = self.backward_augmented(t, y, a)

                # compute g(t-dt)
                dg = torch.autograd.grad(
                    a, self.options.params, y, allow_unused=True, retain_graph=True
                )
                dg = none_to_zeros_like(dg, self.options.params)
                g = add_tuples(g, dg)

            # free the graph of y and a
            y, a = y.data, a.data

            # update progress bar
            self.pbar.update(self.dt)

        # save final augmented state to the solver
        return y, a, g

    @abstractmethod
    def backward_augmented(
        self, t: float, y: Tensor, a: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass
