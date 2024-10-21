from __future__ import annotations

from ..core.abstract_integrator import SMESolveIntegrator
from ..core.diffrax_integrator import (
    EulerIntegrator,
    MilsteinIntegrator,
    SMEDiffraxIntegrator,
)
from ..core.save_mixin import SMESolveSaveMixin


class SMESolveDiffraxIntegrator(
    SMEDiffraxIntegrator, SMESolveIntegrator, SMESolveSaveMixin
):
    """Integrator computing the time evolution of the diffusive SME using the
    Diffrax library."""


# fmt: off
# ruff: noqa
class SMESolveEulerIntegrator(SMESolveDiffraxIntegrator, EulerIntegrator): pass
class SMESolveMilsteinIntegrator(SMESolveDiffraxIntegrator, MilsteinIntegrator): pass
# fmt: on
