from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp

from ...utils.utils import dag
from ..core.abstract_integrator import MESolveIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)


class MESolveDiffraxIntegrator(DiffraxIntegrator, MESolveIntegrator):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation for a single loss channel is:
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

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)

        return dx.ODETerm(vector_field)


class MESolveEulerIntegrator(MESolveDiffraxIntegrator, EulerIntegrator):
    pass


class MESolveDopri5Integrator(MESolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MESolveDopri8Integrator(MESolveDiffraxIntegrator, Dopri8Integrator):
    pass


class MESolveTsit5Integrator(MESolveDiffraxIntegrator, Tsit5Integrator):
    pass


class MESolveKvaerno3Integrator(MESolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MESolveKvaerno5Integrator(MESolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
