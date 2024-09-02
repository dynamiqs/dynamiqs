from __future__ import annotations

from ..core.abstract_integrator import MEPropagatorIntegrator
from ..core.expm_integrator import MEExpmIntegrator, PropagatorExpmIntegrator


class MEPropagatorExpmIntegrator(
    PropagatorExpmIntegrator, MEExpmIntegrator, MEPropagatorIntegrator
):
    pass
