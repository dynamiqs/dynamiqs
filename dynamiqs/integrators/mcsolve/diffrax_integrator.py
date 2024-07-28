from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ...utils.utils import dag
from ..core.abstract_integrator import MCSolveIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)


class MCSolveDiffraxIntegrator(DiffraxIntegrator, MCSolveIntegrator):

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t: Scalar, state: PyTree, _args: PyTree) -> PyTree:
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(axis=0)
            new_state = -1j * (self.H(t) - 1j * 0.5 * LdL) @ state
            return new_state
        return dx.ODETerm(vector_field)

    @property
    def event(self):
        def norm_below_rand(t, y, *args, **kwargs):
            prob = jnp.abs(jnp.einsum("id,id->", jnp.conj(y), y))
            return prob - self.rand
        return dx.Event(norm_below_rand, self.root_finder)


class MCSolveEulerIntegrator(MCSolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MCSolveDopri5Integrator(MCSolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MCSolveDopri8Integrator(MCSolveDiffraxIntegrator, Dopri8Integrator):
    pass


class MCSolveTsit5Integrator(MCSolveDiffraxIntegrator, Tsit5Integrator):
    pass


class MCSolveKvaerno3Integrator(MCSolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MCSolveKvaerno5Integrator(MCSolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
