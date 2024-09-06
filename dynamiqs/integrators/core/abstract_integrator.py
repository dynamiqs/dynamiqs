from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ..._utils import _concatenate_sort
from ...gradient import Gradient
from ...options import Options
from ...result import (
    MEPropagatorResult,
    MESolveResult,
    PropagatorSaved,
    Result,
    Saved,
    SEPropagatorResult,
    SESolveResult,
    SolveSaved,
)
from ...solver import Solver
from ...time_array import TimeArray
from ...utils.quantum_utils import expect


class AbstractIntegrator(eqx.Module):
    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseIntegrator(AbstractIntegrator):
    ts: Array
    y0: Array
    H: TimeArray
    solver: Solver
    gradient: Gradient | None
    options: Options

    @property
    def t0(self) -> Scalar:
        return self.ts[0] if self.options.t0 is None else self.options.t0

    @property
    def t1(self) -> Scalar:
        return self.ts[-1]

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts

    @abstractmethod
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        pass

    def postprocess_saved(self, saved: Saved, ylast: Array) -> Saved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )
        return saved

    @abstractmethod
    def save(self, y: PyTree) -> Saved:
        pass


class SolveIntegrator(BaseIntegrator):
    Es: Array

    def save(self, y: PyTree) -> Saved:
        ysave = y if self.options.save_states else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        Esave = expect(self.Es, y) if self.Es is not None else None
        return SolveSaved(ysave, extra, Esave)

    def postprocess_saved(self, saved: Saved, ylast: Array) -> Saved:
        saved = super().postprocess_saved(saved, ylast)
        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave.swapaxes(-1, -2))
        return saved


class PropagatorIntegrator(BaseIntegrator):
    def save(self, y: PyTree) -> Saved:
        ysave = y if self.options.save_states else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        return PropagatorSaved(ysave, extra)


class MEIntegrator(BaseIntegrator):
    Ls: list[TimeArray]

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class SESolveIntegrator(SolveIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return SESolveResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )


class MESolveIntegrator(SolveIntegrator, MEIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return MESolveResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )


class SEPropagatorIntegrator(PropagatorIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return SEPropagatorResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )


class MEPropagatorIntegrator(PropagatorIntegrator, MEIntegrator):
    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return MEPropagatorResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )
