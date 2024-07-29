from __future__ import annotations

import diffrax as dx

from ..core.abstract_integrator import SESolveIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)


class SESolveDiffraxIntegrator(DiffraxIntegrator, SESolveIntegrator):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class SESolveEulerIntegrator(SESolveDiffraxIntegrator, EulerIntegrator):
    pass


class SESolveDopri5Integrator(SESolveDiffraxIntegrator, Dopri5Integrator):
    pass


class SESolveDopri8Integrator(SESolveDiffraxIntegrator, Dopri8Integrator):
    pass


class SESolveTsit5Integrator(SESolveDiffraxIntegrator, Tsit5Integrator):
    pass


class SESolveKvaerno3Integrator(SESolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class SESolveKvaerno5Integrator(SESolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
