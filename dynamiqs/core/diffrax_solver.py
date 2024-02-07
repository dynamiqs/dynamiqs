from __future__ import annotations

import warnings
from collections import namedtuple

import diffrax as dx

# from equinox import AbstractVar
from jaxtyping import PyTree

from .._utils import bexpect
from ..gradient import Adjoint, Autograd
from ..result import Result
from .abstract_solver import BaseSolver

SolverArgs = namedtuple('SolverArgs', ['save_states', 'Es'])


def save_fn(_t, y, args: SolverArgs):
    res = {}
    if args.save_states:
        res['ysave'] = y
    if args.Es is not None and len(args.Es) > 0:
        res['Esave'] = bexpect(args.Es, y)
    return res


class DiffraxSolver(BaseSolver):
    diffrax_solver: dx.AbstractSolver
    stepsize_controller: dx.AbstractAdaptiveStepSizeController
    dt0: float | None
    max_steps: int
    term: dx.ODETerm

    def __init__(self, *args):
        super().__init__(*args)

    def run(self) -> PyTree:
        # todo: remove once complex support is stabilized in diffrax
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            # === prepare diffrax arguments
            args = SolverArgs(save_states=self.options.save_states, Es=self.Es)
            saveat = dx.SaveAt(ts=self.ts, fn=save_fn)

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
                t0=self.ts[0],
                t1=self.ts[-1],
                dt0=self.dt0,
                y0=self.y0,
                args=args,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
            )

        # === collect results
        ysave = solution.ys.get('ysave', None)
        Esave = solution.ys.get('Esave', None)
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)

        return Result(self.ts, self.solver, self.gradient, self.options, ysave, Esave)


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
