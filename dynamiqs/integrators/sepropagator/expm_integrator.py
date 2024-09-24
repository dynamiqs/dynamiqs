from __future__ import annotations

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.expm_integrator import SEExpmIntegrator


class SEPropagatorExpmIntegrator(SEExpmIntegrator, SEPropagatorIntegrator):
    """Integrator computing the propagator of the Lindblad master equation by
    explicitly exponentiating the propagator.
    """
