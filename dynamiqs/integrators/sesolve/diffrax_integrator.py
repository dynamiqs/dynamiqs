from __future__ import annotations

from ..core.abstract_integrator import SESolveIntegrator
from ..core.diffrax_integrator import (
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    SEDiffraxIntegrator,
    Tsit5Integrator,
)
from ..core.save_mixin import SolveSaveMixin


class SESolveDiffraxIntegrator(SEDiffraxIntegrator, SESolveIntegrator, SolveSaveMixin):
    """Integrator computing the time evolution of the Schrödinger equation using the
    Diffrax library."""


# fmt: off
# ruff: noqa
class SESolveEulerIntegrator(SESolveDiffraxIntegrator, EulerIntegrator): pass
class SESolveDopri5Integrator(SESolveDiffraxIntegrator, Dopri5Integrator): pass
class SESolveDopri8Integrator(SESolveDiffraxIntegrator, Dopri8Integrator): pass
class SESolveTsit5Integrator(SESolveDiffraxIntegrator, Tsit5Integrator): pass
class SESolveKvaerno3Integrator(SESolveDiffraxIntegrator, Kvaerno3Integrator): pass
class SESolveKvaerno5Integrator(SESolveDiffraxIntegrator, Kvaerno5Integrator): pass
# fmt: on
