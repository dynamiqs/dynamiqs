from __future__ import annotations

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.expm_integrator import PropagatorExpmIntegrator, SEExpmIntegrator


class SEPropagatorExpmIntegrator(
    PropagatorExpmIntegrator, SEExpmIntegrator, SEPropagatorIntegrator
):
    pass
