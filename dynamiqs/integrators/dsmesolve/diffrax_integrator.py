from __future__ import annotations

from ..core.abstract_integrator import DSMESolveIntegrator
from ..core.diffrax_integrator import (
    EulerIntegrator,
    MilsteinIntegrator,
    DSMEDiffraxIntegrator,
)
from ..core.save_mixin import DSMESolveSaveMixin


class DSMESolveDiffraxIntegrator(
    DSMEDiffraxIntegrator, DSMESolveIntegrator, DSMESolveSaveMixin
):
    """Integrator computing the time evolution of the diffusive SME using the
    Diffrax library."""


# fmt: off
# ruff: noqa
class DSMESolveEulerIntegrator(DSMESolveDiffraxIntegrator, EulerIntegrator): pass
class DSMESolveMilsteinIntegrator(DSMESolveDiffraxIntegrator, MilsteinIntegrator): pass
# fmt: on
