from __future__ import annotations

from ..core.abstract_integrator import MEPropagatorIntegrator
from ..core.expm_integrator import MEExpmIntegrator


class MEPropagatorExpmIntegrator(MEExpmIntegrator, MEPropagatorIntegrator):
    pass
