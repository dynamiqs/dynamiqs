from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar

from ..time_array import ConstantTimeArray
from .abstract_solver import BaseSolver, MESolver, SESolver


class PropagatorSolver(BaseSolver):
    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: propagator solvers can make different number of steps between
                # batch elements when batching over PWC objects
                return (
                    f'avg. {self.nsteps.mean()} steps | infos shape'
                    f' {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def __init__(self, *args):
        super().__init__(*args)

        # check that Hamiltonian is time-independent
        if not isinstance(self.H, ConstantTimeArray):
            raise TypeError(
                'Solver `Propagator` requires a time-independent Hamiltonian.'
            )

        # extract the constant array from the `ConstantTimeArray` object
        self.H = self.H.x

    def run(self) -> PyTree:
        # === solve differential equation
        def propagate(y, delta_t):  # noqa: ANN001, ANN202
            # propagate forward except if delta_t is zero
            y = jax.lax.cond(delta_t == 0, lambda: y, lambda: self.forward(delta_t, y))
            # save result
            res = self.save(y)
            return y, res

        # we use `jnp.asarray(self.t0)` because of the bug fixed here:
        # https://github.com/google/jax/pull/19381 (fixed in jax-0.4.24)
        # the `.reshape(-1)` covers the case where `self.t0` is a 0-dimensional array
        delta_ts = jnp.diff(self.ts, prepend=jnp.asarray(self.t0).reshape(-1))
        ylast, saved = jax.lax.scan(propagate, self.y0, delta_ts)

        # === collect and return results
        nsteps = (delta_ts != 0).sum()
        saved = self.collect_saved(saved, ylast)
        return self.result(saved, infos=self.Infos(nsteps))

    @abstractmethod
    def forward(self, delta_t: Scalar, y: Array) -> Array:
        pass


class SEPropagatorSolver(PropagatorSolver, SESolver):
    pass


class MEPropagatorSolver(PropagatorSolver, MESolver):
    def __init__(self, *args):
        super().__init__(*args)

        # check that jump operators are time-independent
        if not all(isinstance(L, ConstantTimeArray) for L in self.Ls):
            raise TypeError(
                'Solver `Propagator` requires time-independent jump operators.'
            )

        # extract the constant arrays from the `ConstantTimeArray` objects
        self.Ls = [L.x for L in self.Ls]
