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
    phases = jnp.sort(jnp.angle(jnp.linalg.eigvals(jax.lax.stop_gradient(U))))
    gaps = jnp.diff(phases, append=phases[..., :1] + 2 * jnp.pi)
    widest = jnp.argmax(gaps, axis=-1, keepdims=True)
    middle = jnp.take_along_axis(phases + 0.5 * gaps, widest, axis=-1)
    W = jnp.exp(1j * (jnp.pi - middle))[..., None] * U

    n = U.shape[-1]
    I = jnp.eye(n, dtype=U.dtype)
    # On CPU, OpenBLAS solves with many right-hand sides on several threads. These
    # threads keep spinning after the solve and slow down the next computation. For
    # n < 64, solving 8 columns at a time keeps OpenBLAS on one thread (measured with
    # OpenBLAS 0.3.30 on x86). For larger n, `eigvals` and `eigh` start the threads
    # anyway.
    solve_on_cpu = _solve_by_blocks if n < 64 else jnp.linalg.solve
    K = 1j * jax.lax.platform_dependent(
        I + W, I - W, cpu=solve_on_cpu, default=jnp.linalg.solve
    )
    _, evecs = jnp.linalg.eigh(0.5 * (K + K.mT.conj()))

    # fix the arbitrary phase of each mode by making its largest component positive
    largest = jnp.argmax(jnp.abs(evecs), axis=-2, keepdims=True)
    pivots = jnp.take_along_axis(evecs, largest, axis=-2)
    evecs = evecs * pivots.conj() / jnp.abs(pivots)

    evals = (evecs.conj() * (U @ evecs)).sum(-2)
    return evals, evecs


def _solve_by_blocks(A: Array, B: Array, size: int = 8) -> Array:
    # solve AX = B for `size` columns of B at a time, with one LU factorization of A
    lu_and_piv = jax.scipy.linalg.lu_factor(A)
    n = B.shape[-1]
    B = jnp.pad(B, [(0, 0)] * (B.ndim - 1) + [(0, -n % size)])
    blocks = jnp.moveaxis(B.reshape(*B.shape[:-1], -1, size), -2, 0)
    X = jax.lax.map(lambda b: jax.scipy.linalg.lu_solve(lu_and_piv, b), blocks)
    return jnp.moveaxis(X, 0, -2).reshape(B.shape)[..., :n]


floquet_integrator_constructor = SEFloquetIntegrator
