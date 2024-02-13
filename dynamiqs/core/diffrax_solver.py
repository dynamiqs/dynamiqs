from __future__ import annotations

import warnings

import diffrax as dx
from jaxtyping import PyTree

from ..gradient import Adjoint, Autograd
from .abstract_solver import BaseSolver


class DiffraxSolver(BaseSolver):
    diffrax_solver: dx.AbstractSolver
    stepsize_controller: dx.AbstractAdaptiveStepSizeController
    dt0: float | None
    max_steps: int
    term: dx.ODETerm

    def __init__(self, *args):
        # this dummy init is needed because of the way the class hierarchy is set up,
        # to have subsequent init working properly
        super().__init__(*args)

    def run(self) -> PyTree:
        # todo: remove once complex support is stabilized in diffrax
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            # === prepare diffrax arguments
            fn = lambda t, y, args: self.save(y)
            saveat = dx.SaveAt(ts=self.ts, fn=fn)

            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, Adjoint):
                adjoint = dx.BacksolveAdjoint()

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.term,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.ts[-1],
                dt0=self.dt0,
                y0=self.y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
            )

        # === collect and return results
        saved = solution.ys
        return self.result(saved)


class EulerSolver(DiffraxSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.diffrax_solver = dx.Euler()
        self.stepsize_controller = dx.ConstantStepSize()
        self.dt0 = self.solver.dt
        self.max_steps = 100_000  # todo: fix hard-coded max_steps


class AdaptiveSolver(DiffraxSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.stepsize_controller = dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
        )
        self.dt0 = None
        self.max_steps = self.solver.max_steps


class Dopri5Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri5()


class Dopri8Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri8()


class Tsit5Solver(AdaptiveSolver):
    diffrax_solver = dx.Tsit5()
