# The vmap/batching infrastructure and public API design in this module are adapted
# from work by Derek Everett (https://github.com/derekeverett) on the mesteadystate
# solver for dynamiqs.

from __future__ import annotations

import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import dynamiqs as dq
from dynamiqs._checks import check_hermitian, check_qarray_is_dense, check_shape
from dynamiqs.options import Options, check_options
from dynamiqs.qarrays.qarray import QArray, QArrayLike
from dynamiqs.qarrays.utils import asqarray

from ...integrators._utils import cartesian_vmap, catch_xla_runtime_error, multi_vmap
from ..linear_system.gmres import gmres
from ..preconditionner.lyapunov_solver import LyapunovSolverEig
from .utils import (
    finalize_density_matrix,
    from_dm,
    from_matrix,
    to_dm,
    to_matrix,
    update_preconditioner,
)


# === Result type ===


class GMRESAuxInfo(eqx.Module):
    """Auxiliary information returned by the GMRES steady-state solver.

    Attributes:
        n_iteration: Number of outer GMRES iterations performed.
        success: Whether the solver converged within the specified tolerance.
        recycling: Recycled Krylov vectors `(U, C)` that can be reused in
            subsequent solves.
    """

    n_iteration: int
    success: Array | bool
    recycling: tuple[Array, Array]


class SteadyStateGMRESResult(eqx.Module):
    """Result of the GMRES steady-state solver.

    Attributes:
        rho: The steady-state density matrix, of shape `(..., n, n)`.
        infos: Auxiliary solver information (`GMRESAuxInfo`).
    """

    rho: QArray
    infos: GMRESAuxInfo

    @staticmethod
    def out_axes():
        return SteadyStateGMRESResult(
            rho=0, infos=GMRESAuxInfo(n_iteration=0, success=0, recycling=(0, 0))
        )


# === Public API ===


def steadystate(
    H: QArrayLike,
    jump_ops: list[QArrayLike],
    *,
    rho0: QArrayLike | None = None,
    tol: float = 1e-4,
    max_iteration: int = 100,
    krylov_size: int = 32,
    recycling: int = 5,
    exact_dm: bool = True,
    norm_type: str = 'max',
    options: Options = Options(),  # noqa: B008
) -> SteadyStateGMRESResult:
    r"""Compute the steady state of the Lindblad master equation using GMRES.

    This function finds the density matrix $\rho_\mathrm{ss}$ satisfying
    $$
        \mathcal{L}(\rho_\mathrm{ss}) = -i[H, \rho_\mathrm{ss}]
        + \sum_{k=1}^N \left(
            L_k \rho_\mathrm{ss} L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho_\mathrm{ss}
            - \frac{1}{2} \rho_\mathrm{ss} L_k^\dag L_k
        \right) = 0.
    $$

    The steady state is computed using a preconditioned GMRES algorithm. The
    Lindbladian is deflated with a rank-1 update to enforce the trace constraint,
    and the resulting linear system is preconditioned by a Lyapunov equation solver.

    Note:
        This function supports batched computation over `H`, `jump_ops`, and `rho0`
        via JAX's `vmap`, as well as gradient computation via implicit
        differentiation.

    Args:
        H *(qarray of shape (..., n, n))*: Hamiltonian.
        jump_ops *(list of qarray, each of shape (..., n, n))*: Jump operators.
        rho0 *(qarray of shape (..., n, n), optional)*: Initial guess for the density
            matrix. Defaults to `None`, which uses the vacuum state
            $|0\rangle\langle 0|$.
        tol: Tolerance for the stopping criterion. The solver stops when
            $\|\mathcal{L}(\rho)\| < \mathrm{tol}$, where the norm is determined
            by `norm_type`. Defaults to `1e-4`.
        max_iteration: Maximum number of outer GMRES iterations. Defaults to `100`.
        krylov_size: Size of the Krylov subspace used in each GMRES restart cycle.
            Defaults to `32`. Can be increased to `64` or `128` if convergence is
            slow.
        recycling: Number of Krylov vectors to recycle between restarts.
            Defaults to `5`.
        exact_dm: If `True`, the final density matrix is projected onto the set of
            valid density matrices (positive semi-definite with unit trace). If
            `False`, only Hermitization and trace normalization are applied.
            Defaults to `True`.
        norm_type: Norm used in the stopping criterion. Supported values:
            `'max'` (element-wise max) and `'norm2'` (Frobenius norm).
            Defaults to `'max'`.
        options: Generic dynamiqs solver options (e.g. `cartesian_batching`).

    Returns:
        `SteadyStateGMRESResult` with fields:

        - **`rho`** *(qarray of shape (..., n, n))* — The steady-state density matrix.
        - **`infos`** *(`GMRESAuxInfo`)* — Auxiliary solver information containing
          `n_iteration`, `success`, and `recycling`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a.dag() @ a
        jump_ops = [a]

        result = dq.steadystate(H, jump_ops)
        print(f'Converged: {result.infos.success}')
        print(f'Iterations: {result.infos.n_iteration}')
        ```
    """
    supported_norm_types = ('max', 'norm2')
    if norm_type not in supported_norm_types:
        raise ValueError(
            f'Unknown norm type {norm_type!r}. Expected one of {supported_norm_types}.'
        )

    # === convert arguments ===
    H = asqarray(H)
    Ls = [asqarray(L) for L in jump_ops]
    if rho0 is not None:
        rho0 = asqarray(rho0)

    # === check arguments ===
    _check_steadystate_args(H, Ls, rho0)
    check_options(options, 'steadystate')
    options = options.initialise()

    # === convert rho0 to density matrix ===
    if rho0 is not None:
        rho0 = rho0.todm()
        rho0 = check_hermitian(rho0, 'rho0')

    return _vectorized_steadystate(
        H,
        Ls,
        rho0,
        tol,
        max_iteration,
        krylov_size,
        recycling,
        exact_dm,
        norm_type,
        options,
    )


# === Vectorized entry point ===


@catch_xla_runtime_error
@partial(
    jax.jit,
    static_argnames=(
        'tol',
        'max_iteration',
        'krylov_size',
        'recycling',
        'exact_dm',
        'norm_type',
        'options',
    ),
)
def _vectorized_steadystate(
    H: QArray,
    Ls: list[QArray],
    rho0: QArray | None,
    tol: float,
    max_iteration: int,
    krylov_size: int,
    recycling: int,
    exact_dm: bool,
    norm_type: str,
    options: Options,
) -> SteadyStateGMRESResult:
    # Build in_axes: vectorize over H, Ls, rho0 (if provided).
    # QArrays don't have .in_axes (that's a TimeQArray attribute), so we use
    # 0 if the array has batch dimensions, None otherwise.

    if options is None:
        options = Options(cartesian_batching=False)

    def _qarray_in_axes(q: QArray) -> int | None:
        return 0 if q.ndim > 2 else None

    H_in_axes = _qarray_in_axes(H)
    Ls_in_axes = [_qarray_in_axes(L) for L in Ls]
    rho0_in_axes = _qarray_in_axes(rho0) if rho0 is not None else None
    in_axes = (
        H_in_axes,
        Ls_in_axes,
        rho0_in_axes,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    out_axes = SteadyStateGMRESResult.out_axes()

    if options.cartesian_batching:
        rho0_nvmap = rho0.ndim - 2 if rho0 is not None else 0
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0_nvmap, 0, 0, 0, 0, 0, 0, 0)
        f = cartesian_vmap(_steadystate, in_axes, out_axes, nvmap)
    else:
        arrays_to_broadcast = [H, *Ls]
        if rho0 is not None:
            arrays_to_broadcast.append(rho0)
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in arrays_to_broadcast])
        nvmap = len(bshape)

        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        if rho0 is not None:
            rho0 = rho0.broadcast_to(*bshape, n, n)

        f = multi_vmap(_steadystate, in_axes, out_axes, nvmap)

    return f(
        H,
        Ls,
        rho0,
        tol,
        max_iteration,
        krylov_size,
        recycling,
        exact_dm,
        norm_type,
        options,
    )


# === Core solver (single instance, no batching) ===


def _steadystate(
    H: QArray,
    Ls: list[QArray],
    rho0: QArray | None,
    tol: float,
    max_iteration: int,
    krylov_size: int,
    recycling: int,
    exact_dm: bool,
    norm_type: str,
    options: Options,
) -> SteadyStateGMRESResult:
    hilbert_size = H.shape[-1]
    hilbert_dimensions = H.dims

    # === Build G matrix and preconditioner ===
    Ls_q = dq.stack(Ls)
    LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()
    G = 1j * H.to_jax() + 0.5 * LdagL
    dtype = G.dtype

    # The eigen decomposition is not differentiable, but we use implicit
    # differentiation through custom_linear_solve, so stop_gradient is safe here.
    preconditioner = LyapunovSolverEig(jax.lax.stop_gradient(G))

    # === Initial guess ===
    if rho0 is None:
        rho0 = dq.coherent_dm(hilbert_size, 0.0)
    x_0 = from_dm(rho0)

    # === Deflated linear system: (L + |I><I|) vec(rho) = vec(I) ===
    identity_vectorized = from_matrix(jnp.eye(hilbert_size, dtype=dtype))
    rhs = identity_vectorized

    def lindbladian(x: Array) -> Array:
        return from_dm(
            dq.lindbladian(H, Ls_q, to_dm(x, n=hilbert_size, dims=hilbert_dimensions))
        )

    def lindbladian_plus_rank1(x: Array) -> Array:
        return lindbladian(x) + identity_vectorized.dot(x) * identity_vectorized

    # === Preconditioner with Sherman-Morrison correction ===
    def base_preconditioner(x: Array) -> Array:
        return -from_matrix(preconditioner.solve(to_matrix(x, n=hilbert_size), mu=0.0))

    preconditioner_fn = update_preconditioner(
        base_preconditioner, identity_vectorized, use_rank_1_update=True
    )

    # === Stopping criterion ===
    def stopping_criterion(x: Array) -> Array:
        x_mat = to_matrix(x, hilbert_size)
        x_mat = 0.5 * (x_mat.conj().mT + x_mat)
        x_mat = x_mat / jnp.trace(x_mat)
        lind_x = lindbladian(from_matrix(x_mat))
        if norm_type == 'max':
            norm = jnp.max(jnp.abs(lind_x))
        else:  # norm_type == 'norm2'
            norm = jnp.linalg.norm(lind_x)
        return norm < tol

    # === Solve ===
    x, (n_iteration, success, U, C) = gmres(
        lindbladian_plus_rank1,
        preconditioner_fn,
        x_0,
        rhs,
        stopping_criterion,
        max_iteration,
        krylov_size,
        recycling,
    )

    # === Finalize density matrix ===
    rho = finalize_density_matrix(to_matrix(x, n=hilbert_size), exact_dm)
    rho = to_dm(from_matrix(rho), n=hilbert_size, dims=hilbert_dimensions)

    infos = GMRESAuxInfo(n_iteration=n_iteration, success=success, recycling=(U, C))

    return SteadyStateGMRESResult(rho=rho, infos=infos)


# === Argument checking ===


def _check_steadystate_args(H: QArray, Ls: list[QArray], rho0: QArray | None):
    # === check H shape ===
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape ===
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        warnings.warn(
            'Argument `jump_ops` is an empty list. The Lindbladian reduces to '
            'unitary evolution, which has no unique steady state unless the '
            'Hamiltonian is zero.',
            stacklevel=3,
        )

    # === check rho0 shape and layout ===
    if rho0 is not None:
        check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})
        check_qarray_is_dense(rho0, 'rho0')
