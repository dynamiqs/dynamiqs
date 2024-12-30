from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from ...qarrays.qarray import QArray
from ...result import PropagatorSaved, Saved, SolveSaved
from ...utils.general import expect
from .interfaces import OptionsInterface


class AbstractSaveMixin(OptionsInterface):
    """Mixin to assist integrators with data saving."""

    @abstractmethod
    def save(self, y: PyTree) -> Saved:
        ysave = y if self.options.save_states else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        return Saved(ysave, extra)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )
        return saved


class PropagatorSaveMixin(AbstractSaveMixin):
    """Mixin to assist integrators computing propagators with data saving."""

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y)
        return PropagatorSaved(saved.ysave, saved.extra)


class SolveSaveMixin(AbstractSaveMixin):
    """Mixin to assist integrators computing time evolution with data saving."""

    Es: list[QArray]

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y)
        if self.Es is not None:
            Esave = jnp.stack([expect(E, y) for E in self.Es])
        else:
            Esave = None
        return SolveSaved(saved.ysave, saved.extra, Esave)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast)
        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave.swapaxes(-1, -2))
        return saved
