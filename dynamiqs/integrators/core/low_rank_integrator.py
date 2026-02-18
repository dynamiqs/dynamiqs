from __future__ import annotations

from functools import partial
from typing import Literal

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import PyTree

from ...qarrays.utils import asqarray
from ...result import SolveSaved
from .diffrax_integrator import DiffraxIntegrator
from .interfaces import MEInterface, SolveInterface
from .save_mixin import SolveSaveMixin


def qr_solve(A: Array, B: Array) -> Array:
    operator = lx.MatrixLinearOperator(A)

    def solve_single(b: Array) -> Array:
        return lx.linear_solve(operator, b, throw=False, solver=lx.QR()).value

    return jax.vmap(solve_single, in_axes=1, out_axes=1)(B)  # vmap over columns of B


def cholesky_solve(A: Array, B: Array) -> Array:
    A_pinv = jax.scipy.linalg.solve(A.conj().T @ A, A.conj().T, assume_a='pos')
    return A_pinv @ B


def normalize_m(m: Array) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.abs(m) ** 2, axis=(-2, -1), keepdims=True))
    return m / norm


def rho_from_m(m: Array) -> Array:
    rho = m @ m.conj().swapaxes(-2, -1)
    tr = jnp.trace(rho, axis1=-2, axis2=-1)
    return rho / tr[..., None, None]


def initialize_m0_from_ket(
    psi0: Array, M: int, *, eps: float, key: Array | None
) -> Array:
    """Initialize the low-rank d.m. m0 from a pure state |psi0>.

    The first column of m0 is (a rescaled) psi0, and the remaining M-1 columns
    are small random vectors orthogonal to psi0, scaled by `eps`.
    This ensures the initial density matrix rho0 = m0 @ m0† ≈ |psi0><psi0|
    while making the rank M for numerical stability.
    """
    # psi0 has shape (n, 1)
    if key is None:
        key = jax.random.PRNGKey(0)

    n = psi0.shape[0]
    psi0 = psi0 / jnp.linalg.norm(psi0)
    psi0_unit = psi0

    # Rescale the leading column so that its weight plus the M-1 perturbation
    # columns (each with norm eps) sum to 1 in the trace of rho.
    if M > 1:
        psi0 = psi0_unit * jnp.sqrt(jnp.maximum(1.0 - (M - 1) * (eps**2), 0.0))

    if M == 1:
        return normalize_m(psi0)

    # Generate random complex vectors for the remaining M-1 columns.
    key_r, key_i = jax.random.split(key)
    rand_r = jax.random.normal(key_r, (n, M - 1), dtype=psi0.real.dtype)
    rand_i = jax.random.normal(key_i, (n, M - 1), dtype=psi0.real.dtype)
    rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

    # Project out the psi0 component to make perturbations orthogonal to psi0,
    # then QR-orthonormalize them before scaling by eps.
    rand = rand - psi0_unit @ (psi0_unit.conj().T @ rand)
    q, _ = jnp.linalg.qr(rand, mode='reduced')
    m0 = jnp.concatenate([psi0, q * eps], axis=1)
    return normalize_m(m0)


def initialize_m0_from_dm(
    rho0: Array, M: int, *, eps: float, key: Array | None, eigval_tol: float = 1e-12
) -> Array:
    """Initialize the low-rank d.m. m0 from a density matrix rho0.

    The M largest eigenvectors of rho0 are used as columns of m0, scaled by
    the square root of their eigenvalues. Columns corresponding to
    near-zero eigenvalues (below `eigval_tol`) are replaced by small random
    perturbations scaled by `eps`, ensuring that the rank is at least M.
    """
    # Eigendecompose rho0 and keep the M largest eigenvalues/vectors.
    evals, evecs = jnp.linalg.eigh(rho0)
    evals = jnp.maximum(evals, 0.0)
    evals_M = evals[-M:][::-1]
    cols = evecs[:, -M:][:, ::-1] * jnp.sqrt(evals_M)[None, :]

    # For columns with near-zero eigenvalues, add small random perturbations
    # scaled by eps to avoid degeneracies during solve.
    if eps > 0.0:
        if key is None:
            key = jax.random.PRNGKey(0)
        key_r, key_i = jax.random.split(key)
        rand_r = jax.random.normal(key_r, (rho0.shape[0], M), dtype=cols.real.dtype)
        rand_i = jax.random.normal(key_i, (rho0.shape[0], M), dtype=cols.real.dtype)
        rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

        # Only perturb columns whose eigenvalue is below the tolerance.
        tol = eigval_tol * jnp.maximum(jnp.max(evals_M), 1.0)
        mask = (evals_M <= tol).astype(cols.real.dtype)
        cols = cols + rand * (eps * mask)[None, :]

    return normalize_m(cols)


def expval_from_m(m: Array, op: Array) -> Array:
    return jnp.sum(jnp.conj(m) * (op @ m))


class MESolveLowRankDiffraxIntegrator(
    DiffraxIntegrator, MEInterface, SolveSaveMixin, SolveInterface
):
    """Implementation of low-rank method from Goutte, Savona (2025) arxiv:2508.18114.

    Original implementation in Julia:
    https://github.com/leogoutte/low_rank/blob/main/src/low_rank.jl
    """

    save_lowrank_representation_only: bool = eqx.field(static=True)
    linear_solver: Literal['cholesky', 'qr'] = eqx.field(static=True)
    dims: tuple[int, ...] | None = eqx.field(static=True)

    @property
    def terms(self) -> dx.AbstractTerm:
        solve = cholesky_solve if self.linear_solver == 'cholesky' else qr_solve

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            m = y

            H = self.H(t)
            Ls = [L(t) for L in self.Ls]
            dm = (-1j) * (H @ m).to_jax()

            if len(Ls) > 0:
                for L in Ls:
                    Lm = (L @ m).to_jax()
                    tmp = solve(m, Lm)
                    dm = dm + 0.5 * (Lm @ tmp.conj().T) - 0.5 * (L.dag() @ Lm).to_jax()

            return dm

        return dx.ODETerm(vector_field)

    def _rho_from_m(self, m: Array):  # noqa: ANN202
        return asqarray(rho_from_m(m), dims=self.dims)

    def save(self, y: PyTree) -> SolveSaved:
        m = normalize_m(y)
        save_lowrank_representation_only = self.save_lowrank_representation_only

        need_rho = (
            self.options.save_states and not save_lowrank_representation_only
        ) or self.options.save_extra is not None
        rho = self._rho_from_m(m) if need_rho else None

        ysave = None
        if self.options.save_states:
            ysave = (
                asqarray(m, dims=self.dims) if save_lowrank_representation_only else rho
            )
        extra = self.options.save_extra(rho) if self.options.save_extra else None

        if self.Es is not None:
            Esave = jnp.stack([expval_from_m(m, E) for E in self.Es])
        else:
            Esave = None

        return SolveSaved(ysave=ysave, extra=extra, Esave=Esave)

    def postprocess_saved(self, saved: SolveSaved, ylast: PyTree) -> SolveSaved:
        if not self.options.save_states:
            mlast = normalize_m(ylast)
            ylast_save = (
                asqarray(mlast, dims=self.dims)
                if self.save_lowrank_representation_only
                else self._rho_from_m(mlast)
            )
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast_save, is_leaf=lambda x: x is None
            )

        return self.reorder_Esave(saved)


mesolve_lr_euler_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
mesolve_lr_dopri5_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
mesolve_lr_dopri8_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
mesolve_lr_tsit5_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
mesolve_lr_kvaerno3_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
mesolve_lr_kvaerno5_integrator_constructor = partial(
    MESolveLowRankDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)
