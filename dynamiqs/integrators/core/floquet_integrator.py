from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import PyTree

from dynamiqs.result import FloquetSaved

from ..apis.sepropagator import _sepropagator
from .abstract_integrator import BaseFloquetIntegrator
from .interfaces import SEInterface

__all__ = ['FloquetIntegrator']


class FloquetIntegrator(BaseFloquetIntegrator, SEInterface):
    def run(self) -> PyTree:
        # compute propagators for all times at once, with the last being one period
        ts = jnp.append(self.ts, self.t0 + self.T)
        seprop_result = _sepropagator(
            self.H, ts, solver=self.solver, gradient=self.gradient, options=self.options
        )

        # diagonalize the final propagator to get the Floquet modes at t=t0
        evals, evecs = seprop_result.final_propagator._eig()

        # extract quasienergies
        # minus sign and divide by T to account for e^{-i\epsilon T}
        quasienergies = jnp.angle(-evals) / self.T
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega = 2.0 * jnp.pi / self.T
        quasienergies = jnp.mod(quasienergies + 0.5 * omega, omega) - 0.5 * omega

        # propagate the Floquet modes to all times in tsave
        propagators = seprop_result.propagators[:-1, :, :]
        modes = propagators @ evecs  # (ntsave, n, n) @ (n, m) = (ntsave, n, m)
        modes = modes.mT[..., None]  # (ntsave, m, n, 1)
        modes = modes * jnp.exp(
            1j * quasienergies[:, None, None] * self.ts[:, None, None, None]
        )

        # save the Floquet modes and quasienergies
        saved = FloquetSaved(ysave=modes, extra=None, quasienergies=quasienergies)
        return self.result(saved, infos=seprop_result.infos)


floquet_integrator_constructor = FloquetIntegrator
