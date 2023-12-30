from __future__ import annotations

from abc import abstractmethod

import diffrax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .abstract import IterativeSolver, SolverState


def toreal(x: Array) -> Array:
    return jnp.stack((x.real, x.imag), axis=-1)


def tocomplex(x: Array) -> Array:
    return x[..., 0] + 1j * x[..., 1]


class DiffraxSolverState(SolverState):
    def __init__(self, y0: PyTree, result: PyTree):
        self.y = y0
        self.result = result
        # diffrax solver state


class DiffraxSolver(IterativeSolver):
    @property
    @abstractmethod
    def solver(self) -> diffrax.AbstractSolver:
        pass

    @property
    @abstractmethod
    def terms(self):  # todo: typing
        pass

    @property
    def dt0(self) -> float | None:
        return None

    @property
    def stepsize_controller(self) -> diffrax.AbstractAdaptiveStepSizeController:
        return diffrax.ConstantStepSize()

    @property
    @abstractmethod
    def args(self):  # todo: typing
        pass

    def step(self, t0: Array, t1: Array, solver_state: SolverState) -> SolverState:
        # todo: use stepsize_controller
        # todo: pass solver_state to next step() call
        terms = self.terms
        solver = self.solver
        tprev = t0
        tnext = t0 + self.dt0
        y = toreal(solver_state.y)
        args = self.args
        solver_state = solver.init(terms, tprev, tnext, y, args)

        while tprev < t1:
            y, _, _, solver_state, _ = solver.step(
                terms, tprev, tnext, y, args, solver_state, made_jump=False
            )
            tprev = tnext
            tnext = min(tprev + self.dt0, t1)

        y = tocomplex(y)
        return y
