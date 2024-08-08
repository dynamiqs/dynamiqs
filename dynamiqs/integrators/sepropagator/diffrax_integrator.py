from __future__ import annotations

from ..core.abstract_integrator import SEPropagatorIntegrator
from ..core.diffrax_integrator import (
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    SEDiffraxIntegrator,
    Tsit5Integrator,
)


class SEPropagatorDiffraxIntegrator(SEDiffraxIntegrator, SEPropagatorIntegrator):
    pass


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
