from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

# norm below which a new Krylov vector is a (happy) breakdown
_BREAKDOWN = 1e-12


def gmres(
    A: Callable[[Array], Array],
    M: Callable[[Array], Array],
    b: Array,
    x0: Array,
    stop: Callable[[Array], Array],
    krylov_size: int,
    recycling: int,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    # restarted GMRES for A x = b, right-preconditioned by M ~ A^-1, with harmonic Ritz
    # vectors recycled across cycles as in GCRO-DR (Parks et al. 2006)

    # recycled space smaller than the first Krylov space, and than the full space
    recycling = min(recycling, krylov_size - 1, b.shape[0] - 1)

    # the first cycle builds the recycled space (U, C), with A M U = C
    U = C = jnp.zeros((b.shape[0], 0), b.dtype)
    x, U, C, converged, exhausted = _gmres_cycle(
        A, M, b, x0, U, C, stop, krylov_size, recycling
    )

    def cond_fun(carry):  # noqa: ANN001, ANN202
        niter, _, _, _, converged, exhausted = carry
        return (niter < max_iter) & ~converged & ~exhausted

    def body_fun(carry):  # noqa: ANN001, ANN202
        niter, x, U, C, _, _ = carry
        x, U, C, converged, exhausted = _gmres_cycle(
            A, M, b, x, U, C, stop, krylov_size, recycling
        )
        return niter + 1, x, U, C, converged, exhausted

    carry = (jnp.array(1), x, U, C, converged, exhausted)
    niter, x, _, _, converged, _ = jax.lax.while_loop(cond_fun, body_fun, carry)
    return x, niter, converged


def _gmres_cycle(
    A: Callable[[Array], Array],
    M: Callable[[Array], Array],
    b: Array,
    x: Array,
    U: Array,
    C: Array,
    stop: Callable[[Array], Array],
    krylov_size: int,
    recycling: int,
) -> tuple[Array, Array, Array, Array, Array]:
    p = C.shape[1]
    r = b - A(x)
    V, H, B, beta, breakdown = _arnoldi(lambda v: A(M(v)), C, r, krylov_size)

    # A M [U, V_{m-1}] = [C, V_m] Hbar, with Hbar = [[I, B], [0, H]]
    top = jnp.concatenate([jnp.eye(p, dtype=B.dtype), B], axis=1)
    bottom = jnp.concatenate([jnp.zeros((krylov_size, p), H.dtype), H], axis=1)
    Hbar = jnp.concatenate([top, bottom], axis=0)

    # min ||[C^† r, beta e_1] - Hbar y||, lstsq handles breakdown zero columns
    rhs = jnp.zeros(p + krylov_size, r.dtype)
    rhs = rhs.at[:p].set(C.conj().mT @ r).at[p].set(beta)
    y = jnp.linalg.lstsq(Hbar, rhs)[0]
    x = x + M(U @ y[:p] + V[:, :-1] @ y[p:])

    converged = stop(x)
    # neither a saturated Krylov space nor a NaN iterate can improve
    exhausted = (breakdown | jnp.isnan(x).any()) & ~converged
    U, C = _recycle(Hbar, U, C, V, recycling)
    return x, U, C, converged, exhausted


def _arnoldi(
    AM: Callable[[Array], Array], C: Array, r: Array, krylov_size: int
) -> tuple[Array, Array, Array, Array, Array]:
    # orthonormal basis V of the Krylov space of AM from r, orthogonal to C
    p = C.shape[1]
    v, beta = _normalize(r - C @ (C.conj().mT @ r))
    V = jnp.zeros((r.shape[0], krylov_size), r.dtype).at[:, 0].set(v)
    H = jnp.zeros((krylov_size, krylov_size - 1), r.dtype)
    B = jnp.zeros((p, krylov_size - 1), r.dtype)

    def step(carry, k):  # noqa: ANN001, ANN202
        V, H, B = carry
        w = AM(V[:, k])
        # classical Gram-Schmidt against [C, V], twice for orthogonality
        for _ in range(2):
            c = C.conj().mT @ w
            w = w - C @ c
            h = V.conj().mT @ w
            w = w - V @ h
            B = B.at[:, k].add(c)
            H = H.at[:, k].add(h)
        v, norm = _normalize(w)
        H = H.at[k + 1, k].set(norm)
        V = V.at[:, k + 1].set(v)
        return (V, H, B), None

    (V, H, B), _ = jax.lax.scan(step, (V, H, B), jnp.arange(krylov_size - 1))
    breakdown = H[-1, -1] <= _BREAKDOWN
    return V, H, B, beta, breakdown


def _normalize(w: Array) -> tuple[Array, Array]:
    # w / ||w||, or zero at a breakdown
    norm = jnp.linalg.norm(w)
    is_breakdown = norm <= _BREAKDOWN
    return jnp.where(is_breakdown, 0, w / jnp.where(is_breakdown, 1, norm)), norm


def _recycle(Hbar: Array, U: Array, C: Array, V: Array, k: int) -> tuple[Array, Array]:
    # harmonic Ritz vectors of the k smallest singular values of Hbar, as (U, C) with
    # A M U = C and orthonormal C
    if k == 0:
        return U, C
    _, _, Vh = jnp.linalg.svd(Hbar)
    P = Vh.conj().mT[:, -k:]
    U = jnp.concatenate([U, V[:, :-1]], axis=1) @ P
    C = jnp.concatenate([C, V], axis=1) @ (Hbar @ P)
    C, R = jnp.linalg.qr(C)
    # U <- U R^-1 keeps A M U = C
    U = solve_triangular(R, U.mT, trans='T').mT
    return U, C
