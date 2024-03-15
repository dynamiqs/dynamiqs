from __future__ import annotations

import diffrax as dx

from ..core.abstract_solver import SESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Tsit5Solver,
)


class SEDiffraxSolver(DiffraxSolver, SESolver):
    def __init__(self, *args):
        super().__init__(*args)

        # === define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y

        self.terms = dx.ODETerm(vector_field)


class SEEuler(SEDiffraxSolver, EulerSolver):
    pass


class SEDopri5(SEDiffraxSolver, Dopri5Solver):
    pass


class SEDopri8(SEDiffraxSolver, Dopri8Solver):
    pass


class SETsit5(SEDiffraxSolver, Tsit5Solver):
    pass
