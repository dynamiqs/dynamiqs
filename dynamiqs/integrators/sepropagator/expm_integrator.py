from __future__ import annotations

from jax import Array

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.expm_integrator import SEExpmIntegrator


class SEPropagatorExpmIntegrator(SEExpmIntegrator, SEPropagatorIntegrator):
    pass
