from jax.tree_util import Partial
from typing import Callable

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
from ..time_array import TimeArray
from ..utils.utils import dag


class MonteCarloTerm(dx.ODETerm):
    H: TimeArray  # (n, n)
    Ls: list[TimeArray]  # (nL, n, n)
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def __init__(self, H: TimeArray, Ls: TimeArray):
        self.H = H
        self.Ls = Ls

    def vector_field(self, t: Scalar, state: PyTree, _args: PyTree) -> PyTree:
        Ls = jnp.stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        LdL = (Lsd @ Ls).sum(axis=0)
        psi = state[0:-1]
        r = state[-1][..., None]
        new_state = -1j * (self.H(t) - 1j * 0.5 * LdL) @ psi
        return jnp.concatenate((new_state, r))


class MCDiffraxSolver(DiffraxSolver, MCSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.term = MonteCarloTerm(H=self.H, Ls=self.Ls)

        def norm_below_rand(state, **kwargs):
            psi = jnp.squeeze(state.y[0:-1])
            r = jnp.squeeze(state.y[-1])
            return (jnp.conj(psi) @ psi) < r
        self.discrete_terminating_event = dx.DiscreteTerminatingEvent(Partial(norm_below_rand))


class MCEuler(MCDiffraxSolver, EulerSolver):
    pass


class MCDopri5(MCDiffraxSolver, Dopri5Solver):
    pass


class MCDopri8(MCDiffraxSolver, Dopri8Solver):
    pass


class MCTsit5(MCDiffraxSolver, Tsit5Solver):
    pass
