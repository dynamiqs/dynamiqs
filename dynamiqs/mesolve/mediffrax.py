from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
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
        Ls = jnp.stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        LdL = (Lsd @ Ls).sum(axis=0)
        # drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        #         = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        out = (-1j * self.H(t) - 0.5 * LdL) @ rho + 0.5 * (Ls @ rho @ Lsd).sum(0)
        return out + dag(out)


class MEDiffraxSolver(DiffraxSolver, MESolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.terms = LindbladTerm(H=self.H, Ls=self.Ls)


class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass


class MEDopri8(MEDiffraxSolver, Dopri8Solver):
    pass


class METsit5(MEDiffraxSolver, Tsit5Solver):
    pass
