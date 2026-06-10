from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from ...qarrays.qarray import QArray
from ...result import PropagatorSaved, Saved, SolveSaved
from ...utils.general import expect
from .interfaces import OptionsInterface

_SavedT = TypeVar('_SavedT', bound=Saved)


class AbstractSaveMixin(OptionsInterface, Generic[_SavedT]):
    """Mixin to assist integrators with data saving."""

    @abstractmethod
    def save(self, y: PyTree) -> _SavedT:
        pass

    @abstractmethod
    def postprocess_saved(self, saved: _SavedT, ylast: PyTree) -> _SavedT:
        pass


class PropagatorSaveMixin(AbstractSaveMixin[PropagatorSaved]):
    """Mixin to assist integrators computing propagators with data saving."""

    def save(self, y: PyTree) -> PropagatorSaved:
        ysave = y if self.options.save_propagators else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        return PropagatorSaved(ysave, extra)

    def postprocess_saved(self, saved: PropagatorSaved, ylast: PyTree) -> PropagatorSaved:
        # if save_propagators is False save only last propagator
        if not self.options.save_propagators:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )

        return saved


class SolveSaveMixin(AbstractSaveMixin[SolveSaved]):
    """Mixin to assist integrators computing time evolution with data saving."""

    Es: list[PyTree] | None

    def save(self, y: PyTree) -> SolveSaved:
        ysave = y if self.options.save_states else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        if self.Es is not None:
            Esave = jnp.stack([expect(E, y) for E in self.Es])
        else:
            Esave = None
        return SolveSaved(ysave, extra, Esave)

    def reorder_Esave(self, saved: SolveSaved) -> SolveSaved:
        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave.swapaxes(-1, -2))
        return saved

    def postprocess_saved(self, saved: SolveSaved, ylast: PyTree) -> SolveSaved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )
        return self.reorder_Esave(saved)
