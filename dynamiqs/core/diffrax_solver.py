from __future__ import annotations

import warnings

import diffrax as dx
from jaxtyping import PyTree

from ..gradient import Adjoint, Autograd
from ..utils.utils import expect
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

            if self.gradient is None or isinstance(self.gradient, Autograd):
                adjoint = dx.RecursiveCheckpointAdjoint()
                fn = lambda t, y, args: self.save(y)
                saveat = dx.SaveAt(ts=self.ts, fn=fn)
            elif isinstance(self.gradient, Adjoint):
                if self.options.save_states:
                    adjoint = dx.BacksolveAdjoint()
                    saveat = dx.SaveAt(ts=self.ts)
                else:
                    raise ValueError('Adjoint method requires `options.save_states=True`')

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.term,
                self.diffrax_solver,
                t0=self.ts[0],
                t1=self.ts[-1],
                dt0=self.dt0,
                y0=self.y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
            )

        # === collect and return results
        if self.gradient is None or isinstance(self.gradient, Autograd):
            saved = solution.ys
        elif isinstance(self.gradient, Adjoint):
            saved = {'ysave': solution.ys}
            if self.Es is not None and len(self.Es) > 0:
                expects = expect(self.Es, solution.ys)
                expects = expects.swapaxes(-1, -2)
                saved['Esave'] = expects

        return self.result(saved)


class EulerSolver(DiffraxSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.diffrax_solver = dx.Euler()
        self.stepsize_controller = dx.ConstantStepSize()
        self.dt0 = self.solver.dt
        self.max_steps = 100_000  # todo: fix hard-coded max_steps


class Dopri5Solver(DiffraxSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.diffrax_solver = dx.Dopri5()
        self.stepsize_controller = dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
        )
        self.dt0 = None
        self.max_steps = self.solver.max_steps
