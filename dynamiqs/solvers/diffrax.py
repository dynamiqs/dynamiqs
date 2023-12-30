from __future__ import annotations

from abc import abstractmethod

import diffrax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .abstract import IterativeSolver, SolverState

# todo: rename the file


def toreal(x: Array) -> Array:
    return jnp.stack((x.real, x.imag), axis=-1)


def tocomplex(x: Array) -> Array:
    return x[..., 0] + 1j * x[..., 1]


class DiffraxSolverState(SolverState):
    def __init__(self, y0: PyTree, result: PyTree):
        self.y = y0
        self.result = result
        # todo: Add `diffrax_solver_state` to our solver state.


class DiffraxSolver(IterativeSolver):
    @property
    @abstractmethod
    def diffrax_solver(self) -> diffrax.AbstractSolver:
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
        # todo: Save `diffrax_solver_state` in our `solver_state`, to pass it from one
        #       iteration to the next.
        # todo: Use `stepsize_controller` for adaptive step size.
        terms = self.terms
        diffrax_solver = self.diffrax_solver
        tprev = t0
        tnext = t0 + self.dt0
        y = toreal(solver_state.y)
        args = self.args
        diffrax_solver_state = diffrax_solver.init(terms, tprev, tnext, y, args)

        # todo: this while is not JIT-compatible
        while tprev < t1:
            y, _, _, diffrax_solver_state, _ = diffrax_solver.step(
                terms, tprev, tnext, y, args, diffrax_solver_state, made_jump=False
            )
            tprev = tnext
            tnext = min(tprev + self.dt0, t1)

        solver_state.y = tocomplex(y)
        return solver_state
