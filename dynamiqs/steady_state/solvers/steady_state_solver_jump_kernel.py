from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import dynamiqs as dq

from ...qarrays.qarray import QArray
from ..api.steady_state_solver import Options, SteadyStateResult, SteadyStateSolver
from ..api.utils import finalize_density_matrix, from_matrix, to_dm


class JumpKernelAuxInfo(eqx.Module):
    """Auxiliary information returned by the jump-kernel steady-state solver."""

    nullity: Array
    rank: Array
    success: Array
    steady_norm: Array


class SteadyStateJumpKernelResult(SteadyStateResult):
    """Result of the jump-kernel steady-state solver.

    Attributes:
        rho: The candidate steady-state density matrix, of shape `(..., n, n)`.
        infos: Auxiliary diagnostics (`JumpKernelAuxInfo`).
    """

    rho: QArray
    infos: JumpKernelAuxInfo

    @staticmethod
    def out_axes() -> SteadyStateJumpKernelResult:
        return SteadyStateJumpKernelResult(
            rho=0, infos=JumpKernelAuxInfo(nullity=0, rank=0, success=0, steady_norm=0)
        )


class SteadyStateJumpKernel(SteadyStateSolver):
    r"""Steady-state solver based on the common kernel of jump operators.

    This solver builds a candidate state from the dark subspace of the jump
    operators. Let $\\rho$ satisfy a Lindblad equation with jump operators
    $\\{L_k\\}$. Define the stacked linear map
    $$
        \\mathcal{A} =
        \\begin{bmatrix}
            L_1 \\
            \\vdots \\
            L_m
        \\end{bmatrix},
    $$
    and compute its singular-value decomposition. The right-singular vectors
    corresponding to singular values below `tol` span an approximation of
    $\\bigcap_k \\ker L_k$.

    A projector $P$ onto this subspace is built and converted to a density
    matrix candidate
    $$
        \\rho_0 = \\frac{P}{\\operatorname{tr}(P)}.
    $$
    If no dark component is detected (`tr(P)=0`), the maximally mixed state is
    used as a fallback. In both cases the result is Hermitized and normalized
    (and optionally projected onto valid density matrices with `exact_dm=True`).

    The solver then checks the Lindbladian residual and reports success when
    $$
        \\|\\mathcal{L}(\\rho_0)\\| < \\texttt{steady_tol},
    $$
    with norm selected by `norm_type`.

    Args:
        tol: Singular-value threshold used to detect the nullspace of the
            stacked jump-operator map.
        exact_dm: If `True`, project onto the set of valid density matrices.
        steady_tol: Residual tolerance used in the final steady-state check.
        norm_type: Residual norm used in the final check (`'max'` or `'norm2'`).
    """

    tol: float = 1e-4
    exact_dm: bool = True
    steady_tol: float = 1e-4
    norm_type: str = 'max'

    @staticmethod
    def result_type() -> type[SteadyStateJumpKernelResult]:
        return SteadyStateJumpKernelResult

    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateJumpKernelResult:
        del rho0, options
        return _steadystate_jump_kernel(
            H, Ls, self.tol, self.exact_dm, self.steady_tol, self.norm_type
        )


def _steadystate_jump_kernel(
    H: QArray,
    Ls: list[QArray],
    tol: float,
    exact_dm: bool,
    steady_tol: float,
    norm_type: str,
) -> SteadyStateJumpKernelResult:
    n = H.shape[-1]
    dims = H.dims

    dtype_H = H.to_jax().dtype
    dtype_L = Ls[0].to_jax().dtype if len(Ls) > 0 else dtype_H
    dtype = jnp.result_type(dtype_H, dtype_L)
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128

    if len(Ls) == 0:
        rho_mat = jnp.eye(n, dtype=dtype) / jnp.asarray(n, dtype=dtype)
        rho_mat = finalize_density_matrix(rho_mat, exact_dm)
        rho_q = to_dm(from_matrix(rho_mat), n=n, dims=dims)

        lind_rho = dq.lindbladian(H, dq.stack([]), rho_q)
        lind_mat = lind_rho.to_jax()
        steady_norm = (
            jnp.max(jnp.abs(lind_mat))
            if norm_type == 'max'
            else jnp.linalg.norm(lind_mat)
        )
        infos = JumpKernelAuxInfo(
            nullity=jnp.asarray(n, jnp.int32),
            rank=jnp.asarray(0, jnp.int32),
            success=steady_norm < steady_tol,
            steady_norm=steady_norm,
        )
        return SteadyStateJumpKernelResult(rho=rho_q, infos=infos)

    Ls_q = dq.stack(Ls)

    def _steady_check(rho_q: QArray) -> tuple[Array, Array]:
        lind_rho = dq.lindbladian(H, Ls_q, rho_q)
        lind_mat = lind_rho.to_jax()
        steady_norm = (
            jnp.max(jnp.abs(lind_mat))
            if norm_type == 'max'
            else jnp.linalg.norm(lind_mat)
        )
        return steady_norm, steady_norm < steady_tol

    L_stack = jnp.concatenate([L.to_jax().astype(dtype) for L in Ls], axis=0)
    _, S, Vh = jnp.linalg.svd(L_stack, full_matrices=False)

    keep = tol >= S
    nullity = jnp.sum(keep, dtype=jnp.int32)
    rank = jnp.asarray(S.shape[0], dtype=jnp.int32) - nullity

    V = Vh.conj().T
    V_masked = V * keep.astype(dtype)[None, :]
    P = (V_masked @ V.conj().T).astype(dtype)

    trP = jnp.real(jnp.trace(P))

    def _rho_from_P(_: None) -> Array:
        rho_mat = (P / trP).astype(dtype)
        rho_mat = 0.5 * (rho_mat + rho_mat.conj().T)
        return finalize_density_matrix(rho_mat, exact_dm)

    def _rho_mixed(_: None) -> Array:
        rho_mat = jnp.eye(n, dtype=dtype) / jnp.asarray(n, dtype=dtype)
        return finalize_density_matrix(rho_mat, exact_dm)

    rho_mat = jax.lax.cond(trP > 0, _rho_from_P, _rho_mixed, operand=None)
    rho_q = to_dm(from_matrix(rho_mat), n=n, dims=dims)

    steady_norm, is_steady = _steady_check(rho_q)
    infos = JumpKernelAuxInfo(
        nullity=nullity, rank=rank, success=is_steady, steady_norm=steady_norm
    )
    return SteadyStateJumpKernelResult(rho=rho_q, infos=infos)
