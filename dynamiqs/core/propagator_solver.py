from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar

from ..time_array import ConstantTimeArray
from .abstract_solver import BaseSolver, MESolver


class PropagatorSolver(BaseSolver):
    def __init__(self, *args):
        super().__init__(*args)
        # check that Hamiltonian is time-independent
        if not isinstance(self.H, ConstantTimeArray):
            raise TypeError(
                'Solver `Propagator` requires a time-independent Hamiltonian.'
            )
        self.H = self.H.x

    def run(self) -> PyTree:
        # === solve differential equation
        def f(y, delta_t):
            # propagate forward except if delta_t is zero
            y = jax.lax.cond(delta_t == 0, lambda: y, lambda: self.forward(delta_t, y))
            # save result
            res = self.save(y)
            return y, res

        # todo: fix next line once t0 option is implemented
        delta_ts = jnp.diff(jnp.concatenate((jnp.zeros(1), self.ts)))
        saved = jax.lax.scan(f, self.y0, delta_ts)[-1]

        # === collect and return results
        return self.result(saved)

    @abstractmethod
    def forward(self, delta_t: Scalar, y: Array) -> Array:
        pass


SEPropagatorSolver = PropagatorSolver


class MEPropagatorSolver(MESolver, PropagatorSolver):
    def __init__(self, *args):
        MESolver.__init__(self, *args)
        PropagatorSolver.__init__(self, *args[:-1])
        # check that jump operators are time-independent
        if not all(isinstance(L, ConstantTimeArray) for L in self.Ls):
            raise TypeError(
                'Solver `Propagator` requires time-independent jump operators.'
            )
        self.Ls = [L.x for L in self.Ls]
