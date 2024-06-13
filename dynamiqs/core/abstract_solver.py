from __future__ import annotations

from abc import abstractmethod

from diffrax import Event
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar
from optimistix import AbstractRootFinder

from .._utils import _concatenate_sort
from ..gradient import Gradient
from ..options import Options
from ..result import MEResult, Result, Saved, SEResult, FinalSaved
from ..solver import Solver
from ..time_array import TimeArray
from ..utils.utils import expect, unit


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
        return self.ts[-1] if self.options.t1 is None else self.options.t1

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts

    def save(self, y: PyTree) -> Saved:
        ysave, Esave, extra = None, None, None
        if self.options.save_states:
            ysave = y
        if self.Es is not None and len(self.Es) > 0:
            Esave = expect(self.Es, y)
        if self.options.save_extra is not None:
            extra = self.options.save_extra(y)

        return Saved(ysave, Esave, extra)

    def collect_saved(self, saved: Saved, ylast: Array) -> Saved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )

        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        Esave = saved.Esave
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)
            saved = eqx.tree_at(lambda x: x.Esave, saved, Esave)

        return FinalSaved(saved.ysave, saved.Esave, saved.extra, ylast)

    @abstractmethod
    def result(self, saved: Saved, final_time: float, infos: PyTree | None = None) -> Result:
        pass

    @property
    def event(self):
        return None


class SESolver(BaseSolver):
    def result(self, saved: Saved, final_time: float, infos: PyTree | None = None) -> Result:
        return SEResult(self.ts, self.solver, self.gradient, self.options, saved, final_time, infos)


class MESolver(BaseSolver):
    Ls: list[TimeArray] | None = None

    def result(self, saved: Saved, final_time: float, infos: PyTree | None = None) -> Result:
        return MEResult(self.ts, self.solver, self.gradient, self.options, saved, final_time, infos)


class MCSolver(BaseSolver):
    Ls: list[Array | TimeArray] | None = None
    rand: float = 0.0
    root_finder: AbstractRootFinder | None = None

    def result(self, saved: Saved, final_time: float, infos: PyTree | None = None) -> Result:
        return Result(self.ts, self.solver, self.gradient, self.options, saved, final_time, infos)

    def save(self, y: PyTree) -> Saved:
        return super().save(unit(y))

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)
