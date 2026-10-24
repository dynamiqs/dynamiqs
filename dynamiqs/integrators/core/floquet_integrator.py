from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs.result import FloquetSaved

from ...qarrays.qarray import QArray
from ...qarrays.utils import asqarray
from ...result import FloquetResult, Result
from ..apis.sepropagator import _sepropagator
from .abstract_integrator import BaseIntegrator
from .interfaces import SEInterface


class FloquetIntegrator(BaseIntegrator):
    T: float

    def result(self, saved: FloquetSaved, infos: PyTree | None = None) -> Result:
        return FloquetResult(
            self.ts, self.method, self.gradient, self.options, saved, infos, self.T
        )


class SEFloquetIntegrator(FloquetIntegrator, SEInterface):
    def run(self) -> PyTree:
        # enforce `save_propagators` to be `True` for _sepropagator
        options = eqx.tree_at(lambda opt: opt.save_propagators, self.options, True)

        # compute propagators for all times at once, with the last being one period
        ts = jnp.append(self.ts, self.t0 + self.T)
        seprop_result = _sepropagator(
            self.H, ts, method=self.method, gradient=self.gradient, options=options
        )

        # diagonalize the final propagator to get the Floquet modes at t=t0
        U = seprop_result.final_propagator.to_jax()
        evals, evecs = _diagonalize_unitary(U)

        # extract quasienergies
        # minus sign and divide by T to account for e^{-i\epsilon T}
        quasienergies = -jnp.angle(evals) / self.T
        # quasienergies are only defined modulo 2pi / T. Usual convention is to
        # normalize quasienergies to the region -pi/T, pi/T
        omega = 2.0 * jnp.pi / self.T
        quasienergies = jnp.mod(quasienergies + 0.5 * omega, omega) - 0.5 * omega

        # propagate the Floquet modes to all times in tsave
        propagators = cast(QArray, seprop_result.propagators[:-1, :, :])
        modes = propagators @ evecs  # (ntsave, n, n) @ (n, m) = (ntsave, n, m)
        # convert each mode to a ket of shape (n, 1)
        modes = asqarray(modes.mT[..., None], dims=propagators.dims)
        modes = modes * jnp.exp(
            1j * quasienergies[:, None, None] * self.ts[:, None, None, None]
        )

        # save the Floquet modes and quasienergies
        saved = FloquetSaved(ysave=modes, extra=None, quasienergies=quasienergies)
        return self.result(saved, infos=seprop_result.infos)


def _diagonalize_unitary(U: Array) -> tuple[Array, Array]:
    # A general eigensolver applied to the numerically non-unitary U returns
    # eigenvectors whose overlaps grow as (integration error)/(quasienergy gap), so
    # they are far from orthonormal in near-degenerate clusters. Instead, `eigh`
    # diagonalizes the Hermitian Cayley transform K = i(1 - W)(1 + W)⁻¹ of W = e^{iθ}U,
    # whose eigenvalues tan(φ/2) are injective in the eigenphases φ of W. The global
    # phase θ moves the pole φ = π to the middle of the widest gap in the spectrum of
    # U. A Schur decomposition would also work, but JAX implements it on CPU only and
    # without a differentiation rule.
    I = jnp.eye(U.shape[-1], dtype=U.dtype)

    def cayley(W: Array) -> Array:
        K = 1j * jnp.linalg.solve(I + W, I - W)
        return 0.5 * (K + K.mT.conj())

    # Rough eigenphases of W = e^{i}U are enough to find the widest gap. They are the
    # eigenvalues of the transform of W or of -W, whichever has the smaller norm: its
    # pole is farther from the spectrum. The factor e^{i} moves both poles away from
    # ±1 and ±i, which are eigenvalues of many structured unitaries.
    W = jnp.exp(1j) * U
    V = jax.lax.stop_gradient(W)
    K, K_opposite = cayley(V), cayley(-V)
    opposite = jnp.abs(K_opposite).max((-2, -1)) < jnp.abs(K).max((-2, -1))
    K = jnp.where(opposite[..., None, None], K_opposite, K)
    phases = 2 * jnp.arctan(jnp.linalg.eigvalsh(K))  # of V, or of -V
    phases = jnp.sort(jnp.where(opposite[..., None], phases + jnp.pi, phases))
    gaps = jnp.diff(phases, append=phases[..., :1] + 2 * jnp.pi)
    widest = jnp.argmax(gaps, axis=-1, keepdims=True)
    middle = jnp.take_along_axis(phases + 0.5 * gaps, widest, axis=-1)
    W = jnp.exp(1j * (jnp.pi - middle))[..., None] * W
    _, evecs = jnp.linalg.eigh(cayley(W))

    # fix the arbitrary phase of each mode by making its largest component positive
    largest = jnp.argmax(jnp.abs(evecs), axis=-2, keepdims=True)
    pivots = jnp.take_along_axis(evecs, largest, axis=-2)
    evecs = evecs * pivots.conj() / jnp.abs(pivots)

    evals = (evecs.conj() * (U @ evecs)).sum(-2)
    return evals, evecs


floquet_integrator_constructor = SEFloquetIntegrator
