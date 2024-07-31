from __future__ import annotations

from jax import Array

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.expm_integrator import ExpmIntegrator


class SEPropagatorExpmIntegrator(ExpmIntegrator, SEPropagatorIntegrator):
    def _diff_eq_rhs(self, t: float) -> Array:
        return -1j * self.H(t)
