from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

import dynamiqs as dq

from ...options import Options
from ...qarrays.qarray import QArray
from ..api.steady_state_solver import SteadyStateResult, SteadyStateSolver
from ..api.utils import (
    finalize_density_matrix,
    from_dm,
    from_matrix,
    to_matrix,
    update_preconditioner,
)
from ..preconditionner.lyapunov_solver import LyapunovSolverEig

# =============================================================================
# Optimized Lindbladian operations with explicit derivatives
# =============================================================================


def _lindbladian_matvec(H: jax.Array, Ls: jax.Array, x: jax.Array, n: int) -> jax.Array:
    """Apply vectorized Lindbladian to a vectorized density matrix.

    This computes L(rho) where L is the Lindbladian superoperator:
        L(rho) = -i[H, rho] + sum_k (Lk @ rho @ Lk† - 0.5{Lk†Lk, rho})

    Args:
        H: Hamiltonian matrix (n, n)
        Ls: Stacked jump operators (num_ops, n, n)
        x: Vectorized density matrix (n*n,)
        n: Hilbert space dimension

    Returns:
        Vectorized result of L(rho), shape (n*n,)
    """
    rho = to_matrix(x, n)

    # Commutator: -i[H, rho]
    result = -1j * (H @ rho - rho @ H)

    # Dissipator: sum_k D[Lk](rho)
    for k in range(Ls.shape[0]):
        Lk = Ls[k]
        LkdagLk = Lk.conj().T @ Lk
        result = result + Lk @ rho @ Lk.conj().T - 0.5 * (LkdagLk @ rho + rho @ LkdagLk)

    return from_matrix(result)


def _lindbladian_matvec_transpose(
    H: jax.Array, Ls: jax.Array, x: jax.Array, n: int
) -> jax.Array:
    """Apply the transpose of the vectorized Lindbladian.

    The transpose (not adjoint) of the Lindbladian L is L^T such that
        <y, L(x)> = <L^T(y), x>
    for vectors x, y using the standard inner product (no conjugate):
        <a, b> = sum_i a_i * b_i

    For the Lindbladian:
        L(rho) = -i[H, rho] + sum_k (Lk @ rho @ Lk† - 0.5{Lk†Lk, rho})

    The transpose is:
        L^T(sigma) = i[H^T, sigma] + sum_k (Lk^T @ sigma @ Lk* - 0.5{(Lk†Lk)^T, sigma})

    This is derived from:
        <sigma, L(rho)> = Tr(sigma^T @ L(rho)) = <L^T(sigma), rho>

    Args:
        H: Hamiltonian matrix (n, n)
        Ls: Stacked jump operators (num_ops, n, n)
        x: Vectorized matrix (n*n,)
        n: Hilbert space dimension

    Returns:
        Vectorized result, shape (n*n,)
    """
    sigma = to_matrix(x, n)

    # Transpose of commutator -i[H, rho]: -i[H^T, sigma]
    HT = H.T
    result = -1j * (HT @ sigma - sigma @ HT)

    # Transpose of dissipator
    for k in range(Ls.shape[0]):
        Lk = Ls[k]
        LkT = Lk.T
        Lkconj = Lk.conj()
        LkdagLk = Lk.conj().T @ Lk
        LkdagLk_T = LkdagLk.T
        # Transpose of (Lk @ rho @ Lk†): Lk^T @ sigma @ Lk*
        # Transpose of {Lk†Lk, rho}: {(Lk†Lk)^T, sigma}
        result = result + LkT @ sigma @ Lkconj - 0.5 * (LkdagLk_T @ sigma + sigma @ LkdagLk_T)

    return from_matrix(result)


def _lindbladian_jvp_H(
    dH: jax.Array, rho: jax.Array, n: int
) -> jax.Array:
    """Compute the JVP of Lindbladian with respect to H.

    d(L(H, Ls, rho))/dH @ dH = -i[dH, rho]

    Args:
        dH: Tangent of H (n, n)
        rho: Density matrix (n, n)
        n: Hilbert space dimension

    Returns:
        Vectorized tangent of L(rho), shape (n*n,)
    """
    return from_matrix(-1j * (dH @ rho - rho @ dH))


def _lindbladian_jvp_Ls(
    dLs: jax.Array, Ls: jax.Array, rho: jax.Array, n: int
) -> jax.Array:
    """Compute the JVP of Lindbladian with respect to Ls.

    For each jump operator Lk:
        d(D[Lk](rho))/dLk @ dLk = dLk @ rho @ Lk† + Lk @ rho @ dLk†
                                   - 0.5{dLk†@Lk + Lk†@dLk, rho}

    Args:
        dLs: Tangent of Ls (num_ops, n, n)
        Ls: Jump operators (num_ops, n, n)
        rho: Density matrix (n, n)
        n: Hilbert space dimension

    Returns:
        Vectorized tangent of L(rho), shape (n*n,)
    """
    result = jnp.zeros((n, n), dtype=rho.dtype)

    for k in range(Ls.shape[0]):
        Lk = Ls[k]
        dLk = dLs[k]
        Lkdag = Lk.conj().T
        dLkdag = dLk.conj().T

        # d(Lk @ rho @ Lk†)/dLk = dLk @ rho @ Lk† + Lk @ rho @ dLk†
        d_main = dLk @ rho @ Lkdag + Lk @ rho @ dLkdag

        # d(Lk†Lk @ rho)/dLk = (dLk†@Lk + Lk†@dLk) @ rho
        # d(rho @ Lk†Lk)/dLk = rho @ (dLk†@Lk + Lk†@dLk)
        dLdagL = dLkdag @ Lk + Lkdag @ dLk
        d_anticomm = dLdagL @ rho + rho @ dLdagL

        result = result + d_main - 0.5 * d_anticomm

    return from_matrix(result)


def _lindbladian_vjp_H(
    cotangent_vec: jax.Array, rho: jax.Array, n: int
) -> jax.Array:
    """Compute the VJP of Lindbladian with respect to H.

    For the Lindbladian L(H, Ls, rho) = -i[H, rho] + dissipator,
    the VJP w.r.t. H gives the gradient:
        grad_H = -i(rho @ v† - v† @ rho)
    where v = to_matrix(cotangent_vec).

    This comes from:
        Re(<cotangent, dL/dH @ dH>) = Re(Tr(cotangent† @ (-i[dH, rho])))
                                    = Re(Tr(-i cotangent† @ dH @ rho + i cotangent† @ rho @ dH))
                                    = Re(Tr(dH @ (-i rho @ cotangent† + i cotangent† @ rho)))
                                    = Re(<-i(rho @ v† - v† @ rho), dH>)

    For a complex function, we need the conjugate: grad = -i(rho @ v - v @ rho).conj()

    Args:
        cotangent_vec: Incoming cotangent (n*n,)
        rho: Density matrix (n, n)
        n: Hilbert space dimension

    Returns:
        Gradient w.r.t. H, shape (n, n)
    """
    v = to_matrix(cotangent_vec, n)
    # The gradient with respect to H (Wirtinger derivative for complex H)
    grad_H = -1j * (rho @ v.conj().T - v.conj().T @ rho)
    return grad_H


def _lindbladian_vjp_Ls(
    cotangent_vec: jax.Array, Ls: jax.Array, rho: jax.Array, n: int
) -> jax.Array:
    """Compute the VJP of Lindbladian with respect to Ls.

    For each jump operator, we compute:
        grad_Lk = d<cotangent, D[Lk](rho)>/d(Lk*)

    where D[Lk](rho) = Lk @ rho @ Lk† - 0.5{Lk†Lk, rho}

    Args:
        cotangent_vec: Incoming cotangent (n*n,)
        Ls: Jump operators (num_ops, n, n)
        rho: Density matrix (n, n)
        n: Hilbert space dimension

    Returns:
        Gradient w.r.t. Ls, shape (num_ops, n, n)
    """
    v = to_matrix(cotangent_vec, n)
    vdag = v.conj().T
    num_ops = Ls.shape[0]
    grad_Ls = jnp.zeros_like(Ls)

    for k in range(num_ops):
        Lk = Ls[k]
        Lkdag = Lk.conj().T

        # d<v, Lk @ rho @ Lk†>/dLk* = rho @ Lk† @ v† + Lk @ rho† @ v
        #                           = rho @ Lkdag @ vdag (for the Lk term)
        # For complex derivative, we need Wirtinger:
        # grad w.r.t. Lk (conjugate) from Lk @ rho @ Lk†:
        grad_main = vdag @ Lk @ rho + rho @ Lkdag @ vdag

        # From -0.5 * Lk†Lk @ rho - 0.5 * rho @ Lk†Lk:
        # d<v, Lk†Lk @ rho>/dLk* = Lk @ rho @ v† (contribution to grad_Lk)
        # d<v, rho @ Lk†Lk>/dLk* = v† @ rho @ Lk (contribution to grad_Lk)
        grad_anticomm = Lk @ rho @ vdag + vdag @ rho @ Lk

        grad_Lk = grad_main - 0.5 * grad_anticomm
        grad_Ls = grad_Ls.at[k].set(grad_Lk)

    return grad_Ls


# =============================================================================
# Core solve with custom VJP for efficient gradient computation
# =============================================================================


def _make_preconditioner_components(
    H: jax.Array, Ls: jax.Array, n: int, n_refinement: int
) -> tuple:
    """Build the preconditioner components from H and Ls.

    Returns G matrix and helper functions for preconditioning.
    """
    # Compute G = iH + 0.5 * sum_k Lk†Lk
    LdagL = jnp.sum(Ls.conj().swapaxes(-1, -2) @ Ls, axis=0)
    G = 1j * H + 0.5 * LdagL
    return G


def _build_gmres_solver(
    H: jax.Array,
    Ls: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    dtype: jnp.dtype,
):
    """Build GMRES solver components.

    Returns:
        - identity_vec: Vectorized identity matrix
        - precond_fn: Forward preconditioner
        - precond_fn_adj: Adjoint preconditioner
        - deflated_matvec: Deflated Lindbladian matvec
        - deflated_matvec_transpose: Transpose of deflated matvec
    """
    identity_vec = from_matrix(jnp.eye(n, dtype=dtype))
    krylov_size = min(krylov_size, n * n - 1)

    # Build preconditioner (not differentiated)
    G = jax.lax.stop_gradient(_make_preconditioner_components(H, Ls, n, n_refinement))

    lyap_solver = LyapunovSolverEig(G, n_refinement=n_refinement)
    # For transpose of Lindbladian L^T, use G^T (not G^†)
    # L^T(sigma) = -i[H^T, sigma] + ... uses H^T, and
    # G = iH + 0.5 LdagL, so G_T = iH^T + 0.5 (LdagL)^T = G^T
    lyap_solver_adj = LyapunovSolverEig(G.T, n_refinement=n_refinement)

    def precond_base(x: jax.Array) -> jax.Array:
        return -from_matrix(lyap_solver.solve(to_matrix(x, n=n), mu=0.0))

    def precond_base_adj(x: jax.Array) -> jax.Array:
        return -from_matrix(lyap_solver_adj.solve(to_matrix(x, n=n), mu=0.0))

    precond_fn = update_preconditioner(precond_base, identity_vec, use_rank_1_update=True)
    precond_fn_adj = update_preconditioner(
        precond_base_adj, identity_vec, use_rank_1_update=True
    )

    def lindbladian_vec(x: jax.Array) -> jax.Array:
        return _lindbladian_matvec(H, Ls, x, n)

    def lindbladian_vec_transpose(x: jax.Array) -> jax.Array:
        return _lindbladian_matvec_transpose(H, Ls, x, n)

    def deflated_matvec(x: jax.Array) -> jax.Array:
        return lindbladian_vec(x) + identity_vec * jnp.dot(identity_vec, x)

    def deflated_matvec_transpose(x: jax.Array) -> jax.Array:
        return lindbladian_vec_transpose(x) + identity_vec * jnp.dot(identity_vec, x)

    return (
        identity_vec,
        precond_fn,
        precond_fn_adj,
        deflated_matvec,
        deflated_matvec_transpose,
        krylov_size,
    )


def _solve_gmres_forward(
    H: jax.Array,
    Ls: jax.Array,
    x_0: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Forward pass: solve the steady state using GMRES."""
    (
        identity_vec,
        precond_fn,
        _precond_fn_adj,
        deflated_matvec,
        _deflated_matvec_transpose,
        krylov_size,
    ) = _build_gmres_solver(H, Ls, n, tol, max_iter, krylov_size, n_refinement, dtype)

    # Right preconditioning: solve (A @ M^{-1}) y = b, then x = M^{-1} y
    def right_matvec(y: jax.Array) -> jax.Array:
        return deflated_matvec(precond_fn(y))

    y, _info = jax.scipy.sparse.linalg.gmres(
        right_matvec, identity_vec, x0=x_0, tol=tol, restart=krylov_size, maxiter=max_iter
    )
    x_sol = precond_fn(y)

    return x_sol


def _solve_gmres_adjoint(
    H: jax.Array,
    Ls: jax.Array,
    b: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Solve the adjoint linear system for VJP.

    For the forward system A @ x = b with right preconditioning:
        Solve (A @ M^{-1}) y = b, then x = M^{-1} y

    For the adjoint system A^T @ v = c with right preconditioning:
        Solve (A^T @ M_adj^{-1}) y = c, then v = M_adj^{-1} y

    Where M_adj is the preconditioner for A^T (built from G^H instead of G).
    """
    (
        identity_vec,
        _precond_fn,
        precond_fn_adj,
        _deflated_matvec,
        deflated_matvec_transpose,
        krylov_size,
    ) = _build_gmres_solver(H, Ls, n, tol, max_iter, krylov_size, n_refinement, dtype)

    # Right preconditioning: solve (A^T @ M_adj^{-1}) y = b, then v = M_adj^{-1} y
    def right_matvec_adj(y: jax.Array) -> jax.Array:
        return deflated_matvec_transpose(precond_fn_adj(y))

    y, _info = jax.scipy.sparse.linalg.gmres(
        right_matvec_adj,
        b,
        x0=jnp.zeros_like(b),
        tol=tol,
        restart=krylov_size,
        maxiter=max_iter,
    )
    return precond_fn_adj(y)


def _steadystate_solve_core(
    H: jax.Array,
    Ls: jax.Array,
    x_0: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
) -> jax.Array:
    """Core steady state solver - called by both JVP and VJP wrappers."""
    x_sol = _solve_gmres_forward(
        H, Ls, x_0, n, tol, max_iter, krylov_size, n_refinement, dtype
    )
    rho_ss = finalize_density_matrix(to_matrix(x_sol, n=n), exact_dm)
    return rho_ss


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def _steadystate_solve_with_custom_jvp(
    H: jax.Array,
    Ls: jax.Array,
    x_0: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
) -> jax.Array:
    """Solve steady state with custom JVP for efficient forward-mode differentiation."""
    return _steadystate_solve_core(
        H, Ls, x_0, n, tol, max_iter, krylov_size, n_refinement, exact_dm, dtype
    )


@_steadystate_solve_with_custom_jvp.defjvp
def _steadystate_solve_jvp(
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
    primals: tuple,
    tangents: tuple,
) -> tuple[jax.Array, jax.Array]:
    """Custom JVP for steady state solve using implicit function theorem.

    For steady state: L(H, Ls, rho) = 0
    Differentiating: dL/dH @ dH + dL/dLs @ dLs + dL/drho @ drho = 0
    Therefore: drho = -(dL/drho)^{-1} @ (dL/dH @ dH + dL/dLs @ dLs)

    This requires solving a forward linear system instead of autodiff through GMRES.
    """
    H, Ls, x_0 = primals
    dH, dLs, dx_0 = tangents

    # Forward solve - get raw solution and finalized result in one pass
    x_sol = jax.lax.stop_gradient(_solve_gmres_forward(
        H, Ls, x_0, n, tol, max_iter, krylov_size, n_refinement, dtype
    ))
    rho_raw = to_matrix(x_sol, n)
    rho_ss = jax.lax.stop_gradient(finalize_density_matrix(rho_raw, exact_dm))

    # Build solver components (stop gradient on all)
    H_sg = jax.lax.stop_gradient(H)
    Ls_sg = jax.lax.stop_gradient(Ls)

    (
        identity_vec,
        precond_fn,
        _precond_fn_adj,
        deflated_matvec,
        _deflated_matvec_transpose,
        krylov_size_actual,
    ) = _build_gmres_solver(H_sg, Ls_sg, n, tol, max_iter, krylov_size, n_refinement, dtype)

    # Tangent of Lindbladian: dL(rho)/dH @ dH + dL(rho)/dLs @ dLs
    rhs = jnp.zeros(n * n, dtype=dtype)

    # Contribution from dH (if non-zero)
    if not isinstance(dH, jax.custom_derivatives.SymbolicZero):
        rhs = rhs + _lindbladian_jvp_H(dH, rho_raw, n)

    # Contribution from dLs (if non-zero)
    if not isinstance(dLs, jax.custom_derivatives.SymbolicZero):
        rhs = rhs + _lindbladian_jvp_Ls(dLs, Ls_sg, rho_raw, n)

    # Solve: (L + I⊗I) @ drho_vec = -rhs
    # Using right preconditioning: (L @ M^{-1}) @ y = -rhs, then drho = M^{-1} @ y
    def right_matvec(y: jax.Array) -> jax.Array:
        return deflated_matvec(precond_fn(y))

    y, _info = jax.scipy.sparse.linalg.gmres(
        right_matvec,
        -rhs,
        x0=jnp.zeros_like(rhs),
        tol=tol,
        restart=krylov_size_actual,
        maxiter=max_iter,
    )
    drho_vec = precond_fn(y)

    # Apply finalization tangent
    drho_raw = to_matrix(drho_vec, n)

    def finalize_fn(rho_mat):
        return finalize_density_matrix(rho_mat, exact_dm)

    _, drho_ss = jax.jvp(finalize_fn, (rho_raw,), (drho_raw,))

    return rho_ss, drho_ss


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def _steadystate_solve_with_custom_vjp(
    H: jax.Array,
    Ls: jax.Array,
    x_0: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
) -> jax.Array:
    """Solve steady state with custom VJP for efficient gradient computation.

    The key optimization is that we compute gradients using the adjoint method
    with explicit Lindbladian derivatives, avoiding autodiff through GMRES.

    The gradient computation uses the implicit function theorem:
    At steady state: L(H, Ls, rho) = 0
    Differentiating: dL/dparam + dL/drho @ drho/dparam = 0
    Therefore: drho/dparam = -(dL/drho)^{-1} @ dL/dparam

    For VJP with cotangent g:
    grad_param = -g^T @ (dL/drho)^{-1} @ dL/dparam
              = -(L^{-T} @ g)^T @ dL/dparam
              = -v^T @ dL/dparam  where v = L^{-T} @ g
    """
    return _steadystate_solve_core(
        H, Ls, x_0, n, tol, max_iter, krylov_size, n_refinement, exact_dm, dtype
    )


def _steadystate_solve_fwd(
    H: jax.Array,
    Ls: jax.Array,
    x_0: jax.Array,
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
) -> tuple[jax.Array, tuple]:
    """Forward pass for custom VJP."""
    x_sol = _solve_gmres_forward(
        H, Ls, x_0, n, tol, max_iter, krylov_size, n_refinement, dtype
    )
    rho_ss = finalize_density_matrix(to_matrix(x_sol, n=n), exact_dm)
    # Save raw GMRES solution for backward pass (needed for adjoint)
    rho_raw = to_matrix(x_sol, n)
    residuals = (H, Ls, rho_raw, x_0)
    return rho_ss, residuals


def _steadystate_solve_bwd(
    n: int,
    tol: float,
    max_iter: int,
    krylov_size: int,
    n_refinement: int,
    exact_dm: bool,
    dtype: jnp.dtype,
    residuals: tuple,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Backward pass for custom VJP.

    Uses the adjoint method to compute gradients efficiently:
    1. Backprop through finalization: g_raw = d(finalize)/d(rho_raw)^T @ g
    2. Solve adjoint system: (L + I⊗I)^T @ v = g_raw_vec
    3. Compute grad w.r.t. H and Ls using v

    The key insight is the implicit function theorem. For the GMRES solve:
        (L + I⊗I) @ rho_vec = I_vec
    Differentiating w.r.t. parameter p:
        (dL/dp @ rho) + (L + I⊗I) @ (drho/dp) = 0
        => drho/dp = -(L + I⊗I)^{-1} @ (dL/dp @ rho)

    For a scalar loss f(finalize(rho)):
        df/dp = (df/d(finalize)) @ (d(finalize)/drho) @ (drho/dp)
              = -(g_raw)^T @ (L + I⊗I)^{-1} @ (dL/dp @ rho)
              = -v^T @ (dL/dp @ rho)
    where v = (L + I⊗I)^{-T} @ g_raw and g_raw = (d(finalize)/drho)^T @ g
    """
    H, Ls, rho_raw, x_0 = residuals

    # Step 1: Backprop through finalization
    # g is the cotangent w.r.t. finalized rho_ss
    # We need g_raw = gradient w.r.t. raw GMRES output (before finalization)
    def finalize_fn(rho_mat):
        return finalize_density_matrix(rho_mat, exact_dm)

    _, vjp_finalize = jax.vjp(finalize_fn, rho_raw)
    (g_raw,) = vjp_finalize(g)

    # Step 2: Solve adjoint system
    g_raw_vec = from_matrix(g_raw)
    v_vec = _solve_gmres_adjoint(
        H, Ls, g_raw_vec, n, tol, max_iter, krylov_size, n_refinement, dtype
    )

    # Step 3: Compute gradients using the adjoint state
    # grad_p = -<v, dL/dp @ rho_raw>
    rho_vec = from_matrix(rho_raw)

    def lindbladian_fn(H_param, Ls_param):
        return _lindbladian_matvec(H_param, Ls_param, rho_vec, n)

    _, vjp_fn = jax.vjp(lindbladian_fn, H, Ls)
    grad_H, grad_Ls = vjp_fn(-v_vec)

    # No gradient for x_0 (initial guess)
    grad_x0 = jnp.zeros_like(x_0)

    return (grad_H, grad_Ls, grad_x0)


_steadystate_solve_with_custom_vjp.defvjp(_steadystate_solve_fwd, _steadystate_solve_bwd)


class SteadyStateGMRESResult(SteadyStateResult):
    """Result of the GMRES steady-state solver.

    Attributes:
        rho: The steady-state density matrix, of shape `(..., n, n)`.
    """

    rho: QArray

    @staticmethod
    def out_axes() -> SteadyStateGMRESResult:
        return SteadyStateGMRESResult(rho=0)


class SteadyStateGMRES(SteadyStateSolver):
    r"""GMRES steady-state solver.

    Solves the deflated linear system
    $$
        (\mathcal{L} + |I\rangle\langle I|) |x\rangle = |I\rangle
    $$
    via preconditioned GMRES. Differentiation is handled by
    `jax.lax.custom_linear_solve`.

    This solver computes the steady-state density matrix
    $\rho_\infty$ such that $\mathcal{L}(\rho_\infty) = 0$, where
    $\mathcal{L}$ is the Lindbladian superoperator.

    Because $\mathcal{L}$ has a zero eigenvalue (the steady state), the equation
    $\mathcal{L}(\rho)=0$ is singular. We therefore solve a rank-1 *deflated*
    linear system. Using the vectorization $|x\rangle=\mathrm{vec}(\rho)$
    and $|I\rangle=\mathrm{vec}(I)$, we solve
    $$
        \bigl(\mathcal{L} + |I\rangle\langle I|\bigr)
        |x\rangle
        = |I\rangle.
    $$
    The resulting matrix is then Hermitized and trace-normalized to produce
    $\rho_\infty$ (and optionally projected onto the set of valid density
    matrices when `exact_dm=True`).

    Note:
        **Preconditioning and GMRES.** GMRES solves a linear system $Ax=b$ by
        building Krylov subspaces from repeated applications of $A$.
        When $A$ is ill-conditioned, convergence can be slow; a (left)
        preconditioner $M^{-1}$ replaces the system by
        $$
            (M^{-1}A)x = M^{-1}b,
        $$
        ideally making $M^{-1}A$ better conditioned (or more tightly clustered in
        spectrum) so that GMRES reaches the stopping criterion in fewer
        iterations.

    In this solver, the linear system is preconditioned with an operator
    $S^{-1}$ that approximates the inverse of the Lindbladian.

    The preconditioner is built from the Lyapunov part of the Lindbladian.
    Defining
    $$
        G = iH + \tfrac{1}{2}\sum_k L_k^\dagger L_k,
    $$
    the action of $S$ on a matrix $X$ is
    $$
        S(X) = G X + X G^\dagger.
    $$
    the action of $S^{-1}$ on a matrix $Y$ is defined by solving the Lyapunov equation
    $$
        G X + X G^\dagger = Y.
    $$

    Attributes:
        tol: Tolerance for the stopping criterion. The solver stops when
            $\|\mathcal{L}(\rho)\| < \mathrm{tol}$. Defaults to `1e-4`.
        max_iteration: Maximum number of outer GMRES iterations. Defaults to `100`.
        krylov_size: Size of the Krylov subspace used in each GMRES restart cycle.
            Defaults to `96`. Increase to `128` if convergence is slow, decrease to `64`
              or `32` to accelerate convergence if you observe that
              $\|\mathcal{L}(\rho)\| << \mathrm{tol}$ meaning that GMRES is doing more
              work than necessary to reach the desired tolerance.
            Note that increasing `krylov_size` also increases memory usage and
            runtime per iteration.
        exact_dm: If `True`, project the final matrix onto the set of valid density
            matrices (positive semidefinite with unit trace). If `False`, only
            Hermitization and trace normalization are applied. Defaults to `True`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a + a.dag()
        jump_ops = [a]

        # Default parameters
        solver = dq.SteadyStateGMRES()
        result = dq.steadystate(H, jump_ops, solver=solver)

        # Custom parameters for tighter convergence
        solver = dq.SteadyStateGMRES(tol=1e-6, krylov_size=256)
        result = dq.steadystate(H, jump_ops, solver=solver)

        print(result.rho)
        ```
    """

    tol: float = 1e-4
    max_iteration: int = 100
    krylov_size: int = 64
    exact_dm: bool = True
    n_refinement: int = 2
    forward_mode: bool = False  # Use forward-mode (JVP) differentiation instead of VJP

    @staticmethod
    def result_type() -> type[SteadyStateGMRESResult]:
        return SteadyStateGMRESResult

    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateGMRESResult:
        del options

        n = H.shape[-1]
        dims = H.dims
        n_refinement = max(self.n_refinement, 2)

        H_jax = H.to_jax()
        # Stack jump operators into a single array (num_ops, n, n)
        Ls_jax = jnp.stack([L.to_jax() for L in Ls], axis=0)
        dtype = H_jax.dtype

        if rho0 is None:
            rho0 = dq.coherent_dm(n, 0.0)
        x_0 = from_dm(rho0)

        # Use custom_vjp for reverse-mode differentiation (grad)
        # Use custom_jvp for forward-mode differentiation (jvp, jacfwd)
        # By default, use custom_vjp which is more commonly needed for optimization
        if self.forward_mode:
            rho_ss = _steadystate_solve_with_custom_jvp(
                H_jax,
                Ls_jax,
                x_0,
                n,
                self.tol,
                self.max_iteration,
                self.krylov_size,
                n_refinement,
                self.exact_dm,
                dtype,
            )
        else:
            rho_ss = _steadystate_solve_with_custom_vjp(
                H_jax,
                Ls_jax,
                x_0,
                n,
                self.tol,
                self.max_iteration,
                self.krylov_size,
                n_refinement,
                self.exact_dm,
                dtype,
            )

        state_q = dq.asqarray(rho_ss, dims=dims)
        return SteadyStateGMRESResult(rho=state_q)
