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
from ..time_array import ConstantTimeArray
from ..utils.utils import dag


class MEDiffraxSolver(DiffraxSolver, MESolver):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.

        # time-dependent jump operators
        Ls_tdp = [L for L in self.Ls if not isinstance(L, ConstantTimeArray)]

        # static jump operators
        Ls_t_stc = jnp.stack(
            [L.array for L in self.Ls if isinstance(L, ConstantTimeArray)]
        )
        Lsd_t_stc = dag(Ls_t_stc)
        LdL_t_stc = (Lsd_t_stc @ Ls_t_stc).sum(0)

        # non-Hermitian Hamiltonian
        # It is made of the static and time-dependent parts of the Hamiltonian,
        # and of the static jump operators. The time-dependent jump operators
        # are added later in the vector field.
        Hnh = -1j * self.H - 0.5 * LdL_t_stc

        if len(Ls_tdp) == 0:

            def vector_field(t, y, _):  # noqa: ANN001, ANN202
                tmp = Hnh(t) @ y + 0.5 * (Ls_t_stc @ y @ Lsd_t_stc).sum(0)
                return tmp + dag(tmp)
        else:

            def vector_field(t, y, _):  # noqa: ANN001, ANN202
                Ls_t_tdp = jnp.stack([L(t) for L in Ls_tdp])
                Lsd_t_tdp = dag(Ls_t_tdp)
                LdL_t_tdp = (Ls_t_tdp @ Lsd_t_tdp).sum(0)
                Ls_t = jnp.concatenate([Ls_t_stc, Ls_t_tdp])
                Lsd_t = jnp.concatenate([Lsd_t_stc, Lsd_t_tdp])
                tmp = (Hnh(t) - 0.5 * LdL_t_tdp) @ y + 0.5 * (Ls_t @ y @ Lsd_t).sum(0)
                return tmp + dag(tmp)

        return dx.ODETerm(vector_field)


class MEEuler(MEDiffraxSolver, EulerSolver):
    pass


class MEDopri5(MEDiffraxSolver, Dopri5Solver):
    pass


class MEDopri8(MEDiffraxSolver, Dopri8Solver):
    pass


class METsit5(MEDiffraxSolver, Tsit5Solver):
    pass
