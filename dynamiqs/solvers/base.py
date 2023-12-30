from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ..types import Scalar
from ..utils.utils import expect
from .abstract import AbstractSolver, IterativeSolver, SolverState
from .options import Options


class ArraySaver(NamedTuple):
    xs: Array | None
    counter: int = 0

    @classmethod
    def new(cls, shape: tuple[int], dtype) -> ArraySaver:
        # todo: type dtype
        xs = jnp.empty(shape, dtype=dtype)
        return ArraySaver(xs)

    def add(self, x: Array) -> ArraySaver:
        xs = self.xs.at[self.counter].set(x)
        counter = self.counter + 1
        return ArraySaver(xs, counter)


class BaseResult(NamedTuple):
    ys: ArraySaver | None  # saved quantum states (nt, *y0.shape)
    Es: ArraySaver | None  # saved operators expectation values (nt, nE)


class BaseSolver(AbstractSolver):
    def __init__(
        self,
        ts: Array,
        y0: Array,
        H: Array,
        E: Array,
        options: Options,
    ):
        super().__init__(ts, y0)
        self.H = H
        self.E = E
        self.options = options
        self.savey = self.options.save_states
        self.saveE = len(self.E) > 0


class BaseIterativeSolver(BaseSolver, IterativeSolver):
    def init_solver_state(self) -> SolverState:
        # init y
        y = self.y0

        # init result
        shape = (len(self.ts), *self.y0.shape)
        ys = ArraySaver.new(shape, self.y0.dtype) if self.savey else None
        shape = (len(self.ts), len(self.E))
        Es = ArraySaver.new(shape, self.y0.dtype) if self.saveE else None
        result = BaseResult(ys, Es)

        return SolverState(y, result)

    def save(self, t: Scalar, solver_state: SolverState) -> SolverState:
        # update y
        y = solver_state.y

        # update result
        ys, Es = solver_state.result
        if self.savey:
            ys = ys.add(y)
        if self.saveE:
            E = expect(self.E, y)
            Es = Es.add(E)
        result = BaseResult(ys, Es)

        return SolverState(y, result)


SESolver = BaseSolver
SEIterativeSolver = BaseIterativeSolver


class MESolver(BaseSolver):
    def __init__(
        self,
        ts: Array,
        y0: Array,
        H: Array,
        E: Array,
        options: Options,
        L: Array,
    ):
        super().__init__(ts, y0, H, E, options)
        self.L = L


class MEIterativeSolver(MESolver, BaseIterativeSolver):
    pass


class SMEResult(NamedTuple):
    ys: ArraySaver | None  # saved quantum states (ntsave, *y0.shape)
    Es: ArraySaver | None  # saved operators expectation values (ntsave, nE)
    Lms: ArraySaver  # saved measurement results (ntmeas, nLm)


class SMESolver(MESolver):
    def __init__(
        self,
        tyE: Array,
        tLm: Array,
        y0: Array,
        H: Array,
        E: Array,
        options: Options,
        L: Array,
        etas: Array,
    ):
        ts = jnp.unique(jnp.concatenate((tyE, tLm)))
        super().__init__(ts, y0, H, E, options, L)
        self.tyE = tyE
        self.tLm = tLm
        self.etas = etas

        # split jump operators between purely dissipative (eta = 0) and
        # monitored (eta != 0)
        mask = etas == 0.0  # (nL)
        self.Lc = L[mask]  # (nLc, n, n) purely dissipative
        self.Lm = L[~mask]  # (nLm, n, n) monitored
        self.etas = etas[~mask]  # (nLm)


class SMEIterativeSolver(SMESolver, IterativeSolver):
    def init_solver_state(self) -> SolverState:
        # init y
        y = self.y0

        # init result
        shape = (len(self.tyE), *self.y0.shape)
        ys = ArraySaver.new(shape, self.y0.dtype) if self.savey else None
        shape = (len(self.tyE), len(self.E))
        Es = ArraySaver.new(shape, self.y0.dtype) if self.saveE else None
        shape = (len(self.tLm), len(self.Lm))
        Lms = ArraySaver.new(shape, self.y0.dtype)
        result = SMEResult(ys, Es, Lms)

        return SolverState(y, result)

    @abstractmethod
    def get_Lm(self, t: Scalar, solver_state: SolverState) -> Array:
        pass

    def save(self, t: Scalar, solver_state: SolverState) -> SolverState:
        # update y
        y = solver_state.y

        # update result
        ys, Es, Lms = solver_state.result
        if t in self.tyE:
            if self.savey:
                ys = ys.add(y)
            if self.saveE:
                E = expect(self.E, y)
                Es = Es.add(E)
        elif t in self.tLm:
            Lm = self.get_Lm(t, solver_state)
            Lms = Lms.add(Lm)
        result = SMEResult(ys, Es, Lms)

        return SolverState(solver_state.y, result)
