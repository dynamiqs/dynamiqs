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
            self.ts, self.solver, self.gradient, self.options, saved, infos, self.T
        )

    def run(self) -> PyTree:
        options = eqx.tree_at(lambda x: x.save_states, self.options, False)
        U_result = _sepropagator(
            self.H, self.ts, solver=self.solver, gradient=self.gradient, options=options
        )
        evals, evecs = jnp.linalg.eig(U_result.propagators)
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega_d = 2.0 * jnp.pi / self.T
        # minus sign and divide by T to account for e.g. e^{-iHT}
        quasiens = jnp.angle(-evals) / self.T
        quasiens = jnp.where(quasiens > 0.5 * omega_d, quasiens - omega_d, quasiens)
        saved = FloquetSaved(evecs, quasiens)
        return self.result(saved, infos=U_result.infos)


class FloquetIntegrator_t(FloquetIntegrator):
    floquet_modes_0: Array | None
    quasienergies: Array | None

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
        quasiens_t = self.quasienergies[None, None] * self.ts[:, None, None]
        floquet_modes_t = floquet_modes_t * jnp.exp(1j * quasiens_t)
        saved = FloquetSaved(floquet_modes_t, self.quasienergies)
        return self.result(saved, infos=U_result.infos)
