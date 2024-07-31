from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...result import Saved
from ...utils.vectorization import operator_to_vector
from ..core.abstract_integrator import MEPropagatorIntegrator
from ..core.diffrax_integrator import (
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    MEDiffraxIntegrator,
    Tsit5Integrator,
)


class MEPropagatorDiffraxIntegrator(MEDiffraxIntegrator, MEPropagatorIntegrator):
    def collect_saved(self, saved: Saved, ylast: Array) -> Saved:
        def _reshape_to_vec(y: Array) -> Array:
            return jnp.swapaxes(
                operator_to_vector(y)[..., 0, :, :, 0], axis1=-1, axis2=-2
            )

        ylast = _reshape_to_vec(ylast)
        if self.options.save_states:
            states = _reshape_to_vec(saved.ysave)
            saved = eqx.tree_at(lambda x: x.ysave, saved, states)
        return super().collect_saved(saved, ylast)


class MEPropagatorEulerIntegrator(MEPropagatorDiffraxIntegrator, EulerIntegrator):
    pass


class MEPropagatorDopri5Integrator(MEPropagatorDiffraxIntegrator, Dopri5Integrator):
    pass


class MEPropagatorDopri8Integrator(MEPropagatorDiffraxIntegrator, Dopri8Integrator):
    pass


class MEPropagatorTsit5Integrator(MEPropagatorDiffraxIntegrator, Tsit5Integrator):
    pass


class MEPropagatorKvaerno3Integrator(MEPropagatorDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MEPropagatorKvaerno5Integrator(MEPropagatorDiffraxIntegrator, Kvaerno5Integrator):
    pass
