from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from ...result import PropagatorSaved, Saved, SolveSaved
from ...utils.quantum_utils import expect
from .interfaces import OptionsInterface, SolveInterface


class SaveMixin(OptionsInterface):
    """Mixin to assist integrators with data saving."""

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


class PropagatorSaveMixin(SaveMixin):
    """Mixin to assist integrators computing propagators with data saving."""

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y)
        return PropagatorSaved(saved.ysave, saved.extra)


class SolveSaveMixin(SaveMixin, SolveInterface):
    """Mixin to assist integrators computing time evolution with data saving."""

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y)
        Esave = expect(self.Es, y) if self.Es is not None else None
        return SolveSaved(saved.ysave, saved.extra, Esave)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast)
        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave.swapaxes(-1, -2))
        return saved


class SMESolveSaveMixin(SolveSaveMixin):
    """Mixin to assist SME integrators computing time evolution with data saving."""

    def save(self, y: PyTree) -> Saved:
        return super().save(y.rho)
        # return SMESolveSaved(saved.ysave, saved.extra, saved.Esave, y.dYt)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast)

        # Diffrax integrates the state from t0 to t1. In this case, the state is
        # (rho, dYt). So we recover the signal by simply diffing the resulting array.
        Jsave = jnp.diff(saved.Jsave, axis=0)
        Jsave = Jsave.swapaxes(-1, -2)

        # reorder Jsave after jax.lax.scan stacking (ntsave, nLm) -> (nLm, ntsave)
        return eqx.tree_at(lambda x: x.Jsave, saved, Jsave)
