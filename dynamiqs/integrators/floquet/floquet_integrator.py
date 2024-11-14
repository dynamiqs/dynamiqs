from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs.result import FloquetResult, FloquetSaved, Result, Saved

from ...utils.quantum_utils.general import eig_callback_cpu
from ..apis.sepropagator import _sepropagator
from ..core.abstract_integrator import SEIntegrator

__all__ = ['FloquetIntegrator_t0', 'FloquetIntegrator_t']


class FloquetIntegrator_t0(SEIntegrator):
    T: float

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return FloquetResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos, self.T
        )

    def run(self) -> PyTree:
        U_result = _sepropagator(
            self.H,
            jnp.array([self.ts, self.ts + self.T]),
            solver=self.solver,
            gradient=self.gradient,
            options=self.options,
        )
        evals, evecs = eig_callback_cpu(U_result.final_propagator)
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega_d = 2.0 * jnp.pi / self.T
        # minus sign and divide by T to account for e^{-i\epsilon T}
        quasiens = jnp.angle(-evals) / self.T
        quasiens = jnp.where(quasiens > 0.5 * omega_d, quasiens - omega_d, quasiens)
        # want to save floquet modes with shape ijk where i indexes the modes, j has
        # dimension of the Hilbert dim and k has dimension one (encoding they are kets)
        saved = FloquetSaved(ysave=evecs.T[..., None], extra=None, quasiens=quasiens)
        return self.result(saved, infos=U_result.infos)


class FloquetIntegrator_t(FloquetIntegrator_t0):
    floquet_modes_t0: Array | None
    quasienergies: Array | None

    def run(self) -> PyTree:
        U_result = _sepropagator(
            self.H,
            self.ts,
            solver=self.solver,
            gradient=self.gradient,
            options=self.options,
        )
        # f_modes_t0 have indices fjd where f labels each mode, and jd are the
        # components of the mode. Turn it into jf for ease of matmuls
        f_modes_t0 = jnp.squeeze(self.floquet_modes_t0, axis=-1).T
        # floquet_modes_t has indices tjf where t is time
        floquet_modes_t = U_result.propagators @ f_modes_t0
        quasiens_t = self.quasienergies[None, None] * self.ts[:, None, None]
        floquet_modes_t = floquet_modes_t * jnp.exp(1j * quasiens_t)
        # want indices to be in order of tfj
        floquet_modes_t = jnp.transpose(floquet_modes_t, axes=(0, 2, 1))
        saved = FloquetSaved(
            ysave=floquet_modes_t[..., None], extra=None, quasiens=self.quasienergies
        )
        return self.result(saved, infos=U_result.infos)
