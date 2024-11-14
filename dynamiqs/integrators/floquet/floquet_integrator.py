from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import PyTree

from dynamiqs.result import FloquetResult, FloquetSaved, Result, Saved

from ...utils.quantum_utils.general import eig_callback_cpu
from ..apis.sepropagator import _sepropagator
from ..core.abstract_integrator import SEIntegrator

__all__ = ['FloquetIntegrator']


class FloquetIntegrator(SEIntegrator):
    T: float

    RESULT_CLASS = FloquetResult

    def result(self, saved: Saved, infos: PyTree | None = None) -> Result:
        return self.RESULT_CLASS(
            self.ts, self.solver, self.gradient, self.options, saved, infos, self.T
        )

    def run(self) -> FloquetResult:
        # compute propagators for all times at once, with the last being one period
        ts = jnp.append(self.ts, self.t0 + self.T)
        seprop_result = _sepropagator(
            self.H, ts, solver=self.solver, gradient=self.gradient, options=self.options
        )

        # diagonalize the final propagator to get the Floquet modes at t=t0
        evals, evecs = eig_callback_cpu(seprop_result.final_propagator)

        # extract quasienergies
        # minus sign and divide by T to account for e^{-i\epsilon T}
        quasienergies = jnp.angle(-evals) / self.T
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega = 2.0 * jnp.pi / self.T
        quasienergies = jnp.mod(quasienergies + 0.5 * omega, omega) - 0.5 * omega

        # propagate the Floquet modes to all times in tsave
        propagators = seprop_result.propagators[:-1, :, :]
        modes = propagators @ evecs # (ntsave, n, n) @ (n, m) = (ntsave, n, m)
        modes = modes * jnp.exp(1j * quasienergies * self.ts[:, None, None])
        modes = jnp.swapaxes(modes, -1, -2)[..., None] # (ntsave, m, n, 1)

        # save the Floquet modes and quasienergies
        saved = FloquetSaved(ysave=modes, extra=None, quasienergies=quasienergies)
        return self.result(saved, infos=seprop_result.infos)
