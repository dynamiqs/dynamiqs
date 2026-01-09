# Implementation of the low-rank method from Goutte, Savona (2025) arxiv:2508.18114
# Original implementation in Julia: https://github.com/leogoutte/low_rank/blob/main/src/low_rank.jl
from __future__ import annotations

from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import PyTree

from ...qarrays.utils import asqarray
from ...result import LowRankSolveSaved
from .diffrax_integrator import DiffraxIntegrator
from .interfaces import MEInterface, SolveInterface
from .save_mixin import SolveSaveMixin


def lineax_solve(A: Array, B: Array) -> Array:
    operator = lx.MatrixLinearOperator(A)

    def solve_single(b: Array) -> Array:
        x = lx.linear_solve(operator, b, throw=False, solver=lx.QR()).value
        return x

    x = jax.vmap(solve_single, in_axes=1, out_axes=1)(B)
    return x


def normalize_m(m: Array, *, eps: float = 0.0) -> Array:
    norm = jnp.sqrt(jnp.sum(jnp.abs(m) ** 2, axis=(-2, -1), keepdims=True) + eps)
    return m / norm


def rho_from_m(m: Array) -> Array:
    rho = m @ m.conj().swapaxes(-2, -1)
    tr = jnp.trace(rho, axis1=-2, axis2=-1)
    return rho / tr[..., None, None]


def initialize_m0_from_ket(
    psi0: Array, M: int, *, eps: float, key: Array | None
) -> Array:
    psi0 = jnp.asarray(psi0)
    if psi0.ndim == 2 and psi0.shape[1] == 1:
        psi0 = psi0[:, 0]
    if psi0.ndim != 1:
        raise ValueError('psi0 must be a vector of shape (n,) or (n, 1).')

    if key is None:
        key = jax.random.PRNGKey(0)

    psi0 = psi0 / jnp.linalg.norm(psi0)
    psi0_unit = psi0
    if M > 1:
        psi0 = psi0_unit * jnp.sqrt(jnp.maximum(1.0 - (M - 1) * (eps**2), 0.0))

    if M == 1:
        return normalize_m(psi0[:, None])

    key_r, key_i = jax.random.split(key)
    rand_r = jax.random.normal(key_r, (psi0.shape[0], M - 1), dtype=psi0.real.dtype)
    rand_i = jax.random.normal(key_i, (psi0.shape[0], M - 1), dtype=psi0.real.dtype)
    rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)

    rand = rand - psi0_unit[:, None] * (psi0_unit.conj() @ rand)[None, :]
    q, _ = jnp.linalg.qr(rand, mode='reduced')
    m0 = jnp.concatenate([psi0[:, None], q * eps], axis=1)
    return normalize_m(m0)


def initialize_m0_from_dm(
    rho0: Array, M: int, *, eps: float, key: Array | None, eigval_tol: float = 1e-12
) -> Array:
    rho0 = (rho0 + rho0.conj().T) / 2.0
    evals, evecs = jnp.linalg.eigh(rho0)
    evals = jnp.maximum(evals, 0.0)
    evals_M = evals[-M:][::-1]
    cols = evecs[:, -M:][:, ::-1] * jnp.sqrt(evals_M)[None, :]

    if eps > 0.0:
        if key is None:
            key = jax.random.PRNGKey(0)
        key_r, key_i = jax.random.split(key)
        rand_r = jax.random.normal(key_r, (rho0.shape[0], M), dtype=cols.real.dtype)
        rand_i = jax.random.normal(key_i, (rho0.shape[0], M), dtype=cols.real.dtype)
        rand = (rand_r + 1j * rand_i) / jnp.sqrt(2.0)
        tol = eigval_tol * jnp.maximum(jnp.max(evals_M), 1.0)
        mask = (evals_M <= tol).astype(cols.real.dtype)
        cols = cols + rand * (eps * mask)[None, :]

    return normalize_m(cols)


def expval_from_m(m: Array, op: Array) -> Array:
    return jnp.sum(jnp.conj(m) * (op @ m))


def chi_from_m(m: Array) -> Array:
    gram = m.conj().T @ m
    evals = jnp.linalg.eigvalsh(gram)
    return jnp.abs(evals[0] / evals[-1])


class MESolveLowRankDiffraxIntegrator(
    DiffraxIntegrator, MEInterface, SolveSaveMixin, SolveInterface
):
    normalize_each_eval: bool = eqx.field(static=True)
    gram_reg: float = eqx.field(static=True)
    dims: tuple[int, ...] | None = eqx.field(static=True)

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            m = y
            if self.normalize_each_eval:
                m = normalize_m(m, eps=0.0)

            H = self.H(t)
            Ls = [L(t) for L in self.Ls]
            dm = (-1j) * (H @ m).to_jax()

            if len(Ls) > 0:
                for L in Ls:
                    Lm = (L @ m).to_jax()
                    # tmp = m_inv @ Lm
                    tmp = lineax_solve(m, Lm)
                    dm = dm + 0.5 * (Lm @ tmp.conj().T) - 0.5 * (L.dag() @ Lm).to_jax()

            return dm

        return dx.ODETerm(vector_field)

    def _rho_from_m(self, m: Array):  # noqa: ANN202
        return asqarray(rho_from_m(m), dims=self.dims)

    def save(self, y: PyTree) -> LowRankSolveSaved:
        m = normalize_m(y, eps=0.0)
        save_factors_only = self.options.save_factors_only

        rho = None
        if (self.options.save_states and not save_factors_only) or (
            self.options.save_extra is not None
        ):
            rho = self._rho_from_m(m)

        if self.options.save_states:
            ysave = None if save_factors_only else rho
        else:
            ysave = None
        extra = self.options.save_extra(rho) if self.options.save_extra else None

        if self.Es is not None:
            Esave = jnp.stack([expval_from_m(m, E) for E in self.Es])
        else:
            Esave = None

        if self.options.save_low_rank_chi:
            chisave = chi_from_m(m)
        else:
            chisave = None

        if self.options.save_states and save_factors_only:
            msave = m
        else:
            msave = None
        return LowRankSolveSaved(
            ysave=ysave, extra=extra, Esave=Esave, msave=msave, chisave=chisave
        )

    def postprocess_saved(
        self, saved: LowRankSolveSaved, ylast: PyTree
    ) -> LowRankSolveSaved:
        if not self.options.save_states:
            mlast = normalize_m(ylast, eps=0.0)
            if self.options.save_factors_only:
                ylast_save = mlast
            else:
                ylast_save = self._rho_from_m(mlast)
            saved = eqx.tree_at(
                lambda x: x.ysave, saved, ylast_save, is_leaf=lambda x: x is None
            )
            if not self.options.save_factors_only:
                saved = eqx.tree_at(
                    lambda x: x.msave, saved, mlast, is_leaf=lambda x: x is None
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
