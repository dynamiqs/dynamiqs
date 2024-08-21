from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs.result import FloquetResult, FloquetSaved, Result, Saved

from ..apis.sepropagator import _sepropagator
from ..core.abstract_integrator import BaseIntegrator

__all__ = ['FloquetIntegrator']


class FloquetIntegrator(BaseIntegrator):
    T: float

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return FloquetResult(
            None, self.solver, self.gradient, self.options, saved, infos
        )

    def run(self) -> PyTree:
        options = eqx.tree_at(lambda x: x.save_states, self.options, False)
        U_result = _sepropagator(
            self.H, self.ts, solver=self.solver, gradient=self.gradient, options=options
        )
        evals, evecs = jnp.linalg.eig(U_result.propagators)
        # quasi energies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasi energies to the region -pi/T, pi/T
        omega_d = 2.0 * jnp.pi / self.T
        # minus sign and divide by T to account for e.g. e^{-iHT}
        quasi_es = jnp.angle(-evals) / self.T
        quasi_es = jnp.where(quasi_es > 0.5 * omega_d, quasi_es - omega_d, quasi_es)
        saved = FloquetSaved(evecs, quasi_es)
        return self.result(saved, infos=U_result.infos)


class FloquetIntegrator_t(FloquetIntegrator):
    floquet_modes_0: Array | None
    quasi_energies: Array | None

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return FloquetResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )

    def run(self) -> PyTree:
        U_result = _sepropagator(
            self.H,
            self.ts,
            solver=self.solver,
            gradient=self.gradient,
            options=self.options,
        )
        # floquet modes are stored as column vectors, so the multiplication
        # by the phases addresses each column vector individually
        floquet_modes_t = U_result.propagators @ self.floquet_modes_0
        quasi_e_t = self.quasi_energies[None, None] * self.ts[:, None, None]
        floquet_modes_t = floquet_modes_t * jnp.exp(1j * quasi_e_t)
        saved = FloquetSaved(floquet_modes_t, self.quasi_energies)
        return self.result(saved, infos=U_result.infos)
