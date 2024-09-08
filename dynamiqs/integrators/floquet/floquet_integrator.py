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
        U_result = _sepropagator(
            self.H, self.ts, solver=self.solver, gradient=self.gradient, options=self.options
        )
        evals, evecs = jnp.linalg.eig(U_result.final_propagator)
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega_d = 2.0 * jnp.pi / self.T
        # minus sign and divide by T to account for e^{-i\epsilon T}
        quasiens = jnp.angle(-evals) / self.T
        quasiens = jnp.where(quasiens > 0.5 * omega_d, quasiens - omega_d, quasiens)
        # want to save floquet modes with shape ijk where i indexes the modes, j has
        # dimension of the Hilbert dim and k has dimension one (encoding they are kets)
        saved = FloquetSaved(evecs.T[..., None], quasiens)
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
        # floquet_modes_0 have indices fjd where f labels each mode, and jd are the
        # components of the mode. Turn it into jf for ease of matmuls
        f_modes_0 = jnp.squeeze(self.floquet_modes_0, axis=-1).T
        # has indices tjf where t is time
        floquet_modes_t = U_result.propagators @ f_modes_0
        quasiens_t = self.quasienergies[None, None] * self.ts[:, None, None]
        floquet_modes_t = floquet_modes_t * jnp.exp(1j * quasiens_t)
        # want indices to be in order of tfj
        floquet_modes_t = jnp.transpose(floquet_modes_t, axes=(0, 2, 1))
        saved = FloquetSaved(floquet_modes_t[..., None], self.quasienergies)
        return self.result(saved, infos=U_result.infos)
