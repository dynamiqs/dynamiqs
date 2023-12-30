from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ..time_array import TimeArray
from ..types import Scalar
from ..utils.operators import eye
from ..utils.utils import _hdim, dag, expect
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

    def append(self, x: Array) -> ArraySaver:
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
        H: TimeArray,
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
            ys = ys.append(y)
        if self.saveE:
            E = expect(self.E, y)
            Es = Es.append(E)
        result = BaseResult(ys, Es)

        return SolverState(y, result)


SESolver = BaseSolver
SEIterativeSolver = BaseIterativeSolver


class MESolver(BaseSolver):
    def __init__(
        self,
        ts: Array,
        y0: Array,
        H: TimeArray,
        E: Array,
        options: Options,
        L: Array,
    ):
        super().__init__(ts, y0, H, E, options)
        self.L = L  # (nL, n, n)
        self.Ld = dag(L)  # (nL, n, n)
        self.sum_LdL = (self.Ld @ self.L).sum(axis=0)  # (n, n)

        # define the identity operator
        n = _hdim(self.y0)
        self.I = eye(n, dtype=self.options.cdtype)  # (n, n)

    def Hnh(self, t: Scalar) -> Array:
        return self.H(t) - 0.5j * self.sum_LdL  # (n, n)

    def lindbladian(self, t: Scalar, y: Array) -> Array:
        # compute the action of the Lindbladian on a state
        # Note: Hermiticity of the output is enforced to avoid numerical instability
        #       with common ODE solvers.
        tmp = -1j * self.Hnh(t) @ y + 0.5 * (self.L @ y @ self.Ld).sum(0)
        return tmp + dag(tmp)


class MEIterativeSolver(MESolver, BaseIterativeSolver):
    pass


class SMEResult(NamedTuple):
    ys: ArraySaver | None  # saved quantum states (ntsave, *y0.shape)
    Es: ArraySaver | None  # saved operators expectation values (ntsave, nE)
    Lms: ArraySaver  # saved measurement results (ntmeas, nLm)


class SMESolver(MESolver):
    def __init__(
        self,
        ts_yE: Array,
        ts_Lm: Array,
        y0: Array,
        H: TimeArray,
        E: Array,
        options: Options,
        L: Array,
        etas: Array,
    ):
        ts = jnp.unique(jnp.concatenate((ts_yE, ts_Lm)))
        super().__init__(ts, y0, H, E, options, L)
        self.ts_yE = ts_yE
        self.ts_Lm = ts_Lm
        self.etas = etas

        # split jump operators between purely dissipative (eta = 0) and
        # monitored (eta != 0)
        mask = etas == 0.0  # (nL)
        # todo: this masking is not JIT-compatible
        self.Lc = L[mask]  # (nLc, n, n) purely dissipative
        self.Lm = L[~mask]  # (nLm, n, n) monitored
        self.etas = etas[~mask]  # (nLm)


class SMEIterativeSolver(SMESolver, IterativeSolver):
    def init_solver_state(self) -> SolverState:
        # init y
        y = self.y0

        # init result
        shape = (len(self.ts_yE), *self.y0.shape)
        ys = ArraySaver.new(shape, self.y0.dtype) if self.savey else None
        shape = (len(self.ts_yE), len(self.E))
        Es = ArraySaver.new(shape, self.y0.dtype) if self.saveE else None
        shape = (len(self.ts_Lm), len(self.Lm))
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
        # todo: this if is not JIT-compatible
        if t in self.ts_yE:
            if self.savey:
                ys = ys.append(y)
            if self.saveE:
                E = expect(self.E, y)
                Es = Es.append(E)
        elif t in self.ts_Lm:
            Lm = self.get_Lm(t, solver_state)
            Lms = Lms.append(Lm)
        result = SMEResult(ys, Es, Lms)

        return SolverState(solver_state.y, result)
