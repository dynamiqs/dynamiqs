from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from ..gradient import Autograd, CheckpointAutograd
from .abstract_solver import BaseSolver


class DiffraxSolver(BaseSolver):
    diffrax_solver: dx.AbstractSolver
    stepsize_controller: dx.AbstractAdaptiveStepSizeController
    dt0: float | None
    max_steps: int
    terms: dx.AbstractTerm

    def __init__(self, *args):
        # this dummy init is needed because of the way the class hierarchy is set up,
        # to have subsequent init working properly
        super().__init__(*args)

    def run(self) -> PyTree:
        # TODO: remove once complex support is stabilized in diffrax
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            # === prepare diffrax arguments
            fn = lambda t, y, args: self.save(y)  # noqa: ARG005
            subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
            subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
            saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
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
        save_a, save_b = solution.ys
        saved = save_a
        ylast = save_b[0]  # (n, m)
        return self.result(saved, ylast, infos=self.infos(solution.stats))

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedSolver(DiffraxSolver):
    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def __init__(self, *args):
        super().__init__(*args)
        self.stepsize_controller = dx.ConstantStepSize()
        self.dt0 = self.solver.dt
        self.max_steps = 100_000  # TODO: fix hard-coded max_steps

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])


class EulerSolver(FixedSolver):
    diffrax_solver = dx.Euler()


class AdaptiveSolver(DiffraxSolver):
    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean()} steps ({self.naccepted.mean()}'
                    f' accepted, {self.nrejected.mean()} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )

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

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )


class Dopri5Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri5()


class Dopri8Solver(AdaptiveSolver):
    diffrax_solver = dx.Dopri8()


class Tsit5Solver(AdaptiveSolver):
    diffrax_solver = dx.Tsit5()
