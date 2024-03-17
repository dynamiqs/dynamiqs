from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ..gradient import Gradient
from ..options import Options
from ..result import MEResult, Result, Saved, SEResult
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
    H: TimeArray
    Es: Array
    solver: Solver
    gradient: Gradient | None
    options: Options

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    def save(self, y: Array) -> dict[str, Array]:
        saved = {}
        if self.options.save_states:
            saved['ysave'] = y
        if self.Es is not None and len(self.Es) > 0:
            saved['Esave'] = expect(self.Es, y)
        if self.options.save_extra is not None:
            saved['extra'] = self.options.save_extra(y)

        return saved

    def collect_saved(self, saved: dict[str, Array], ylast: Array) -> Saved:
        ysave = saved.get('ysave', ylast)
        Esave = saved.get('Esave')
        extra = saved.get('extra')
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)
        return Saved(ysave, Esave, extra)

    @abstractmethod
    def result(self, saved: Saved, **kwargs) -> Result:
        pass


class SESolver(BaseSolver):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return SEResult(self.ts, self.solver, self.gradient, self.options, saved, infos)


class MESolver(BaseSolver):
    Ls: list[TimeArray]

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return MEResult(self.ts, self.solver, self.gradient, self.options, saved, infos)
