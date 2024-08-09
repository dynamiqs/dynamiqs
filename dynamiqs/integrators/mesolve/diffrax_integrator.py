from __future__ import annotations

from ..core.abstract_integrator import MESolveIntegrator
from ..core.diffrax_integrator import (
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    MEDiffraxIntegrator,
    Tsit5Integrator,
)


class MESolveDiffraxIntegrator(MEDiffraxIntegrator, MESolveIntegrator):
    pass


class MESolveEulerIntegrator(MESolveDiffraxIntegrator, EulerIntegrator):
    pass


class MESolveDopri5Integrator(MESolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MESolveDopri8Integrator(MESolveDiffraxIntegrator, Dopri8Integrator):
    pass


class MESolveTsit5Integrator(MESolveDiffraxIntegrator, Tsit5Integrator):
    pass


class MESolveKvaerno3Integrator(MESolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MESolveKvaerno5Integrator(MESolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
