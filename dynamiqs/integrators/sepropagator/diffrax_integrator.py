from __future__ import annotations

import diffrax as dx

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)


class SEPropagatorDiffraxIntegrator(DiffraxIntegrator, SEPropagatorIntegrator):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term dU/dt = - i H U
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class SEPropagatorEulerIntegrator(SEPropagatorDiffraxIntegrator, EulerIntegrator):
    pass


class SEPropagatorDopri5Integrator(SEPropagatorDiffraxIntegrator, Dopri5Integrator):
    pass


class SEPropagatorDopri8Integrator(SEPropagatorDiffraxIntegrator, Dopri8Integrator):
    pass


class SEPropagatorTsit5Integrator(SEPropagatorDiffraxIntegrator, Tsit5Integrator):
    pass


class SEPropagatorKvaerno3Integrator(SEPropagatorDiffraxIntegrator, Kvaerno3Integrator):
    pass


class SEPropagatorKvaerno5Integrator(SEPropagatorDiffraxIntegrator, Kvaerno5Integrator):
    pass
