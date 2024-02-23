from typing import Callable

import diffrax as dx
from jaxtyping import PyTree, Scalar

from ..core.abstract_solver import SESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Tsit5Solver,
)
from ..time_array import TimeArray


class SchrodingerTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray):
        self.H = H

    def vector_field(self, t: Scalar, psi: PyTree, _args: PyTree) -> PyTree:
        return -1j * self.H(t) @ psi


class SEDiffraxSolver(DiffraxSolver, SESolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.term = SchrodingerTerm(self.H)


class SEEuler(SEDiffraxSolver, EulerSolver):
    pass


class SEDopri5(SEDiffraxSolver, Dopri5Solver):
    pass


class SEDopri8(SEDiffraxSolver, Dopri8Solver):
    pass


class SETsit5(SEDiffraxSolver, Tsit5Solver):
    pass
