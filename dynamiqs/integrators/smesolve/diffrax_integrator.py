from __future__ import annotations

from ..core.abstract_integrator import SMESolveIntegrator
from ..core.diffrax_integrator import (
    EulerIntegrator,
    MilsteinIntegrator,
    SMEDiffraxIntegrator,
)


class SMESolveDiffraxIntegrator(SMEDiffraxIntegrator, SMESolveIntegrator):
    pass


class SMESolveEulerIntegrator(SMESolveDiffraxIntegrator, EulerIntegrator):
    pass


class SMESolveMilsteinIntegrator(SMESolveDiffraxIntegrator, MilsteinIntegrator):
    pass
