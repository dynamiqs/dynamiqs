import jax
from jax.tree_util import Partial

from functools import partial

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ..core.abstract_solver import MCSolver
from ..core.diffrax_solver import (
    DiffraxSolver,
    Dopri5Solver,
    Dopri8Solver,
    EulerSolver,
    Tsit5Solver,
)
from ..utils.utils import dag


class MCDiffraxSolver(DiffraxSolver, MCSolver):

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
    def discrete_terminating_event(self):
        def norm_below_rand(state, **kwargs):
            psi = state.y
            prob = jnp.abs(jnp.einsum("id,id->", jnp.conj(psi), psi))
            return prob < self.rand
        return dx.DiscreteTerminatingEvent(norm_below_rand)


class MCEuler(MCDiffraxSolver, EulerSolver):
    pass


class MCDopri5(MCDiffraxSolver, Dopri5Solver):
    pass


class MCDopri8(MCDiffraxSolver, Dopri8Solver):
    pass


class MCTsit5(MCDiffraxSolver, Tsit5Solver):
    pass
