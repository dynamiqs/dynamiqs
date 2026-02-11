import abc
from collections.abc import Callable
from typing import Any

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..linear_system.gmres import gmres
from ..preconditionner.lyapunov_solver import LyapuSolverEig
from .utils import finalize_density_matrix, dp, update_preconditioner


class GMRESAuxInfo(eqx.Module):
    n_iteration: int
    success: Array | bool
    recycling: tuple[Array, Array]


def steady_state(
    H: dq.QArray,
    Ls: list[dq.QArray],
    tol: float = 1e-8,
    *,
    rank1_coeff: float = 1.0,
    use_rank_1_update: bool = True,
    initial_guess: dq.QArray | None = None,
    max_iter: int = 100,
    krylov_size: int = 32,
    recycling: int = 5,
    exact_dm: bool = False,
    I_normalized: bool = True,
    norm_type: str = 'max',
) -> tuple[dq.QArray, GMRESAuxInfo]:
    """
    Computes the steady state by preconditioning the deflated
    steady-state equation with the lyapunov equation.

    This differs from `solve` with the deflation, which allow to directly enforce the trace and positivity constraint of the steady state.
    """
    n = H.shape[-1]
    dims = H.dims
    Ls_q = dq.stack(Ls)
    LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()
    G = 1j * H.to_jax() + 1 / 2 * LdagL
    dtype = G.dtype

    # The Schur decomposition called in the solver is not differentiable
    # On the other hand, we're differentiating the result of solve with
    # implicit differentiation: we don't need to differentiate through
    # the solver itself. Hence the stop_gradient here.
    preconditioner = LyapuSolverEig(jax.lax.stop_gradient(G))

    def from_matrix(x: Array) -> Array:
        return x.flatten(order='F')

    def to_matrix(x: Array) -> Array:
        return x.reshape((n, n), order='F')

    def to_dm(x: Array) -> dq.QArray:
        return dq.asqarray(to_matrix(x), dims=dims)

    def from_dm(x: dq.QArray) -> Array:
        return from_matrix(x.to_jax())

    if initial_guess is None:
        initial_guess = dq.coherent_dm(n, 0.0)

    x_0 = from_dm(initial_guess)

    if I_normalized:
        rank1_coeff_normalized = rank1_coeff / n**2
    else:
        rank1_coeff_normalized = rank1_coeff

    identity_vectorized = from_matrix(jnp.eye(n, dtype=dtype))
    rhs = rank1_coeff_normalized * identity_vectorized

    def lindbladian(x: Array) -> Array:
        return from_dm(dq.lindbladian(H, Ls_q, to_dm(x)))

    def lindbladian_plus_rank1(x: Array) -> Array:
        return (
            lindbladian(x)
            + rank1_coeff_normalized
            # dot do not perform any conjugation,
            # which is fine since identity is real
            * identity_vectorized.dot(x)
            * identity_vectorized
        )

    def base_preconditioner(x: Array) -> Array:
        return -from_matrix(preconditioner.solve(to_matrix(x), mu=0.0))

    preconditioner_fn = update_preconditioner(
        base_preconditioner,
        identity_vectorized,
        rank1_coeff_normalized,
        use_rank_1_update,
    )

    def stopping_criterion(x: Array) -> Array:
        """Checks if the hermicized, trace-1 density matrix satisfies
        `|| L rho || < tol`."""
        x_mat = to_matrix(x)
        x_mat = 0.5 * (x_mat.conj().mT + x_mat)
        x_mat = x_mat / jnp.trace(x_mat)
        if norm_type == 'max':
            norm = jnp.max(jnp.abs(lindbladian(from_matrix(x_mat))))
        elif norm_type == 'norm2':
            norm = jnp.linalg.norm(lindbladian(from_matrix(x_mat)))
        return norm < tol

    x, (n_iteration, success, U, C) = gmres(
        lindbladian_plus_rank1,
        preconditioner_fn,
        x_0,
        rhs,
        stopping_criterion,
        max_iter,
        krylov_size,
        recycling,
    )
    recycling_info = (U, C)

    rho = finalize_density_matrix(to_matrix(x), exact_dm)
    rho = to_dm(from_matrix(rho))

    return rho, GMRESAuxInfo(n_iteration, success, recycling_info)
