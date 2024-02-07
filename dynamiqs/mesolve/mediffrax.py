from typing import Callable

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from ..core.abstract_solver import MESolver
from ..core.diffrax_solver import DiffraxSolver, Dopri5Solver, EulerSolver
from ..time_array import TimeArray
from ..utils.utils import dag


class LindbladTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: list[TimeArray]  # (nL, n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.H = H
        self.Ls = Ls

    def vector_field(self, t: Scalar, rho: PyTree, _args: PyTree):
        Ls = jnp.stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        Hnh = self.H(t) - 0.5j * (Lsd @ Ls).sum(axis=0)
        out = -1j * Hnh @ rho + 0.5 * (Ls @ rho @ Lsd).sum(0)
        return out + dag(out)


class MEDiffraxSolver(DiffraxSolver, MESolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.term = LindbladTerm(H=self.H, Ls=self.Ls)


class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass
