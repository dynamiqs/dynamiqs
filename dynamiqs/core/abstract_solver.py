from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Solver
from ..time_array import TimeArray
from ..utils.utils import expect


class AbstractSolver(eqx.Module):
    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseSolver(AbstractSolver):
    ts: Array
    y0: Array
    H: Array | TimeArray
    Es: Array
    solver: Solver
    gradient: Gradient | None
    options: Options

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    def save(self, y: Array) -> dict[str, Array]:
        saved = {}
        if self.options.save_states:
            saved['ysave'] = y
        if self.Es is not None and len(self.Es) > 0:
            saved['Esave'] = expect(self.Es, y)
        if self.options.save_fn is not None:
            saved['save'] = self.options.save_fn(y)

        return saved

    def result(self, saved: dict[str, Array], ylast: Array) -> Result:
        ysave = saved.get('ysave', ylast)
        Esave = saved.get('Esave', None)
        save = saved.get('save', None)
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)

        return Result(
            self.ts, self.solver, self.gradient, self.options, ysave, Esave, save
        )


SESolver = BaseSolver


class MESolver(BaseSolver):
    Ls: list[Array | TimeArray]
