from __future__ import annotations

import diffrax as dx
from jaxtyping import PyTree, Scalar

from ..core.abstract_solver import MESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Tsit5Solver,
)
from ..time_array import TimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: list[TimeArray]  # (nL, n, n)
    vector_field: callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.H = H
        self.Ls = Ls

    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree) -> PyTree:
        Ls = [L(t) for L in self.Ls]

        sum_Ldag_L = sum([dag(L) @ L for L in Ls])
        jump_term = sum([L @ rho @ dag(L) for L in Ls])

        out = (-1j * self.H(t) - 0.5 * sum_Ldag_L) @ rho + 0.5 * jump_term

        return out + dag(out)


class MEDiffraxSolver(DiffraxSolver, MESolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.term = LindbladTerm(H=self.H, Ls=self.Ls)


class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass


class MEDopri8(MEDiffraxSolver, Dopri8Solver):
    pass


class METsit5(MEDiffraxSolver, Tsit5Solver):
    pass
