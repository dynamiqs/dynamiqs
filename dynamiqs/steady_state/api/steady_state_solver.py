# The vmap/batching infrastructure and public API design in this module are adapted
# from work by Derek Everett (https://github.com/derekeverett) on the mesteadystate
# solver for dynamiqs.

from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import dynamiqs as dq
from ..._checks import check_hermitian, check_qarray_is_dense, check_shape
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray

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


# =============================================================================
# Auxiliary info types
# =============================================================================


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


# =============================================================================
# Result types
# =============================================================================


class SteadyStateResult(eqx.Module, ABC):
    """Abstract base class for steady-state solver results."""

    rho: QArray
    infos: eqx.Module

    @staticmethod
    @abstractmethod
    def out_axes(): ...


class SteadyStateGMRESResult(SteadyStateResult):
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


# =============================================================================
# Solver classes
# =============================================================================


class SteadyStateSolver(eqx.Module, ABC):
    """Abstract base class for steady-state solvers.

    Subclasses must implement `_run` (the core single-instance solver logic)
    and `result_type` (returning the concrete `SteadyStateResult` subclass).
    """

    @abstractmethod
    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateResult: ...

    @staticmethod
    @abstractmethod
    def result_type() -> type[SteadyStateResult]: ...


class SteadyStateGMRES(SteadyStateSolver):
    r"""GMRES steady-state solver with Krylov recycling.

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

    Krylov subspace recycling is used between restart cycles to reduce the number
    of Lindbladian applications and accelerate convergence.

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

    Args:
        tol: Tolerance for the stopping criterion. The solver stops when
            $\|\mathcal{L}(\rho)\| < \mathrm{tol}$, where the norm is determined
            by `norm_type`. Defaults to `1e-4`.
        max_iteration: Maximum number of outer GMRES iterations. Defaults to `100`.
        krylov_size: Size of the Krylov subspace used in each GMRES restart cycle.
            Defaults to `32`. Increase to `64` or `128` if convergence is slow.
        recycling: Number of Krylov vectors to recycle between restarts.
            Defaults to `5`.
        exact_dm: If `True`, project the final matrix onto the set of valid density
            matrices (positive semidefinite with unit trace). If `False`, only
            Hermitization and trace normalization are applied. Defaults to `True`.
        norm_type: Norm used in the stopping criterion. Supported values:
            `'max'` (element-wise max) and `'norm2'` (Frobenius norm).
            Defaults to `'max'`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a.dag() @ a
        jump_ops = [a]

        # Default parameters
        solver = dq.SteadyStateGMRES()
        result = dq.steadystate(H, jump_ops, solver=solver)

        # Custom parameters for tighter convergence
        solver = dq.SteadyStateGMRES(tol=1e-6, krylov_size=64, exact_dm=False)
        result = dq.steadystate(H, jump_ops, solver=solver)

        print(f'Converged: {result.infos.success}')
        print(f'Iterations: {result.infos.n_iteration}')
        ```
    """

    tol: float = 1e-4
    max_iteration: int = 100
    krylov_size: int = 32
    recycling: int = 5
    exact_dm: bool = True
    norm_type: str = 'max'

    @staticmethod
    def result_type() -> type[SteadyStateGMRESResult]:
        return SteadyStateGMRESResult

    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateGMRESResult:
        return _steadystate_gmres(
            H,
            Ls,
            rho0,
            self.tol,
            self.max_iteration,
            self.krylov_size,
            self.recycling,
            self.exact_dm,
            self.norm_type,
            options,
        )


# =============================================================================
# Public API
# =============================================================================


def steadystate(
    H: QArrayLike,
    jump_ops: list[QArrayLike],
    *,
    rho0: QArrayLike | None = None,
    solver: SteadyStateSolver = SteadyStateGMRES(),  # noqa: B008
    options: Options = Options(),  # noqa: B008
) -> SteadyStateResult:
    r"""Compute the steady state of the Lindblad master equation.

    The Lindblad dynamics are written as
    $$
        \frac{d\rho}{dt} = \mathcal{L}(\rho),
    $$
    with
    $$
        \mathcal{L}(\rho) = -i[H, \rho]
        + \sum_{k=1}^N \left(
            L_k \rho L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho
            - \frac{1}{2} \rho L_k^\dag L_k
        \right).
    $$
    This function finds the steady-state density matrix $\rho_\infty$ such that
    $$
        \mathcal{L}(\rho_\infty) = 0.
    $$

    Note:
        This function supports batched computation over `H`, `jump_ops`, and
        `rho0` via JAX's `vmap`, as well as gradient computation.

    Args:
        H *(qarray of shape (..., n, n))*: Hamiltonian.
        jump_ops *(list of qarray, each of shape (..., n, n))*: Jump operators.
        rho0 *(qarray of shape (..., n, n), optional)*: Initial guess for the
            density matrix. Defaults to `None`, which uses the vacuum state
            $|0\rangle\langle 0|$.
        solver: Solver instance controlling the algorithm and its parameters.
            Defaults to `SteadyStateGMRES()`. See `SteadyStateGMRES` for
            available options.
        options: Generic dynamiqs solver options (e.g. `cartesian_batching`).

    Returns:
        `SteadyStateResult` subclass depending on the solver used. For the
        default `SteadyStateGMRES` solver, returns `SteadyStateGMRESResult`
        with fields:

        - **`rho`** *(qarray of shape (..., n, n))* — The steady-state density
          matrix.
        - **`infos`** *(`GMRESAuxInfo`)* — Auxiliary solver information
          containing `n_iteration`, `success`, and `recycling`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a.dag() @ a
        jump_ops = [a]

        # With default solver
        result = dq.steadystate(H, jump_ops)
        print(f'Converged: {result.infos.success}')
        print(f'Iterations: {result.infos.n_iteration}')

        # With custom solver parameters
        solver = SteadyStateGMRES(tol=1e-6, krylov_size=64)
        result = dq.steadystate(H, jump_ops, solver=solver)
        ```
    """
    # === convert arguments ===
    H = asqarray(H)
    Ls = [asqarray(L) for L in jump_ops]
    if rho0 is not None:
        rho0 = asqarray(rho0)

    # === check arguments ===
    _check_steadystate_args(H, Ls, rho0)
    _check_steadystate_solver(solver)
    check_options(options, 'steadystate')
    options = options.initialise()

    # === convert rho0 to density matrix ===
    if rho0 is not None:
        rho0 = rho0.todm()
        rho0 = check_hermitian(rho0, 'rho0')

    return _vectorized_steadystate(H, Ls, rho0, solver, options)


# =============================================================================
# Vectorized entry point (shared by all solvers)
# =============================================================================


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'options'))
def _vectorized_steadystate(
    H: QArray,
    Ls: list[QArray],
    rho0: QArray | None,
    solver: SteadyStateSolver,
    options: Options,
) -> SteadyStateResult:
    """Vectorized entry point shared by all steady-state solvers.

    Handles batching (flat or cartesian) via `vmap`, then dispatches to
    ``solver._run`` for each individual (non-batched) solve.
    """
    if options is None:
        options = Options(cartesian_batching=False)

    def _qarray_in_axes(q: QArray) -> int | None:
        return 0 if q.ndim > 2 else None

    # --- build in_axes and out_axes ---
    H_in_axes = _qarray_in_axes(H)
    Ls_in_axes = [_qarray_in_axes(L) for L in Ls]
    rho0_in_axes = _qarray_in_axes(rho0) if rho0 is not None else None
    in_axes = (H_in_axes, Ls_in_axes, rho0_in_axes, None)

    out_axes = solver.result_type().out_axes()

    # Closure that captures `solver` (static) and calls its _run method.
    def _run_single(H, Ls, rho0, options):
        return solver._run(H, Ls, rho0, options)

    # --- cartesian batching ---
    if options.cartesian_batching:
        rho0_nvmap = rho0.ndim - 2 if rho0 is not None else 0
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0_nvmap, 0)
        f = cartesian_vmap(_run_single, in_axes, out_axes, nvmap)

    # --- flat (broadcast) batching ---
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

        f = multi_vmap(_run_single, in_axes, out_axes, nvmap)

    return f(H, Ls, rho0, options)


# =============================================================================
# Core GMRES solver (single instance, no batching)
# =============================================================================


def _steadystate_gmres(
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
    """Core GMRES steady-state solver for a single (non-batched) instance."""
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


# =============================================================================
# Argument checking
# =============================================================================


def _check_steadystate_solver(solver: SteadyStateSolver):
    if not isinstance(solver, SteadyStateSolver):
        raise TypeError(
            f'Argument `solver` must be an instance of `SteadyStateSolver`, '
            f'got {type(solver).__name__}.'
        )


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
