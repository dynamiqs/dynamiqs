from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp

from ..core.abstract_solver import MESolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Tsit5Solver,
)
from ..utils.utils import dag


class MEDiffraxSolver(DiffraxSolver, MESolver):
    def __init__(self, *args):
        super().__init__(*args)

        # === define Lindblad term drho/dt
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            # drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
            #         = {(-i H - 0.5 Ld @ L) @ rho + 0.5 L @ rho @ Ld} + h.c.
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(axis=0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)

        self.terms = dx.ODETerm(vector_field)


class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass


class MEDopri8(MEDiffraxSolver, Dopri8Solver):
    pass


class METsit5(MEDiffraxSolver, Tsit5Solver):
    pass
