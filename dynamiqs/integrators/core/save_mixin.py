from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from ...result import DSMESolveSaved, PropagatorSaved, Saved, SolveSaved
from ...utils.general import expect
from .interfaces import OptionsInterface


class AbstractSaveMixin(OptionsInterface):
    """Mixin to assist integrators with data saving."""

    @abstractmethod
    def save(self, y: PyTree) -> Saved:
        pass

    @abstractmethod
    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        pass


class PropagatorSaveMixin(AbstractSaveMixin):
    """Mixin to assist integrators computing propagators with data saving."""

    def save(self, y: PyTree) -> Saved:
        ysave = y if self.options.save_propagators else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        return PropagatorSaved(ysave, extra)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        # if save_propagators is False save only last propagator
        if not self.options.save_propagators:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )

        return saved


class SolveSaveMixin(AbstractSaveMixin):
    """Mixin to assist integrators computing time evolution with data saving."""

    def save(self, y: PyTree) -> Saved:
        ysave = y if self.options.save_states else None
        extra = self.options.save_extra(y) if self.options.save_extra else None
        if self.Es is not None:
            Esave = jnp.stack([expect(E, y) for E in self.Es])
        else:
            Esave = None
        return SolveSaved(ysave, extra, Esave)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        # if save_states is False save only last state
        if not self.options.save_states:
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast, is_leaf=lambda x: x is None
            )

        # reorder Esave after jax.lax.scan stacking (ntsave, nE) -> (nE, ntsave)
        if saved.Esave is not None:
            saved = eqx.tree_at(lambda x: x.Esave, saved, saved.Esave.swapaxes(-1, -2))
        return saved


class DSMESolveSaveMixin(SolveSaveMixin):
    """Mixin to assist diffusive SME integrators computing time evolution with data
    saving.
    """

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y.rho)
        return DSMESolveSaved(saved.ysave, saved.extra, saved.Esave, y.Y)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast.rho)
        # reorder Isave after jax.lax.scan stacking (ntsave, nLm) -> (nLm, ntsave)
        Isave = saved.Isave.swapaxes(-1, -2)
        return eqx.tree_at(lambda x: x.Isave, saved, Isave)
