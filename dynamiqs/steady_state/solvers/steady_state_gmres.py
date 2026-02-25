from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import dynamiqs as dq

from ...options import Options
from ...qarrays.qarray import QArray
from ..api.steady_state_solver import SteadyStateResult, SteadyStateSolver
from ..api.utils import (
    finalize_density_matrix,
    from_dm,
    from_matrix,
    to_dm,
    to_matrix,
    update_preconditioner,
)
from ..linear_system.gmres import gmres
from ..preconditionner.lyapunov_solver import LyapunovSolverEig


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


class SteadyStateGMRESResult(SteadyStateResult):
    """Result of the GMRES steady-state solver.

    Attributes:
        rho: The steady-state density matrix, of shape `(..., n, n)`.
        infos: Auxiliary solver information (`GMRESAuxInfo`).
    """

    rho: QArray
    infos: GMRESAuxInfo

    @staticmethod
    def out_axes() -> SteadyStateGMRESResult:
        return SteadyStateGMRESResult(
            rho=0, infos=GMRESAuxInfo(n_iteration=0, success=0, recycling=(0, 0))
        )


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
            Defaults to `32`. Increase to `64` or `128` if convergence is slow
            in term of iterations.
            Note that increasing `krylov_size` also increases memory usage and runtime
              per iteration.
        recycling: Number of Krylov vectors to recycle between restarts.
            Defaults to `5`. The recycling is necessary if you have convergence issues,
              but it can cause longer overall runtime.
            If you have convergence issues, try increasing `krylov_size` first,
            then add recycling if the number of iterations is still high.
        exact_dm: If `True`, project the final matrix onto the set of valid density
            matrices (positive semidefinite with unit trace). If `False`, only
            Hermitization and trace normalization are applied. Defaults to `True`.
        norm_type: Norm used in the stopping criterion. Supported values:
            `'max'` (element-wise max) and `'norm2'` (Frobenius norm).
            Defaults to `'max'`.
        check_pure_imaginary_eigenvalue: If `True`, perform an early spectral
            validation on
            $G=-iH-\tfrac{1}{2}\sum_k L_k^\dagger L_k$
            and raise an error if at least one eigenvalue has near-zero real
            part (purely imaginary eigenvalue), because GMRES preconditioning
            can fail in this regime.
            Defaults to `False`.
            The tolerance is automatically set to machine epsilon of the
            corresponding real dtype.

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
        solver = dq.SteadyStateGMRES(tol=1e-6, krylov_size=32, exact_dm=True)
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
    check_pure_imaginary_eigenvalue: bool = False

    @staticmethod
    def result_type() -> type[SteadyStateGMRESResult]:
        return SteadyStateGMRESResult

    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateGMRESResult:
        state, (n_iteration, success, U, C) = steadystate_gmres_single_instance(
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
        infos = GMRESAuxInfo(n_iteration=n_iteration, success=success, recycling=(U, C))
        return SteadyStateGMRESResult(rho=state, infos=infos)


def steadystate_gmres_single_instance(
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
) -> tuple[QArray, tuple[Array, Array | bool, Array, Array]]:
    """Core GMRES steady-state solver for a single (non-batched) instance."""
    del options

    hilbert_size = H.shape[-1]
    hilbert_dimensions = H.dims

    Ls_q = dq.stack(Ls)
    LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()
    G = 1j * H.to_jax() + 0.5 * LdagL
    dtype = G.dtype

    preconditioner = LyapunovSolverEig(jax.lax.stop_gradient(G))

    if rho0 is None:
        rho0 = dq.coherent_dm(hilbert_size, 0.0)
    x_0 = from_dm(rho0)

    identity_vectorized = from_matrix(jnp.eye(hilbert_size, dtype=dtype))
    rhs = identity_vectorized

    def lindbladian(x: Array) -> Array:
        return from_dm(
            dq.lindbladian(H, Ls_q, to_dm(x, n=hilbert_size, dims=hilbert_dimensions))
        )

    def lindbladian_plus_rank1(x: Array) -> Array:
        return lindbladian(x) + identity_vectorized.dot(x) * identity_vectorized

    def base_preconditioner(x: Array) -> Array:
        return -from_matrix(preconditioner.solve(to_matrix(x, n=hilbert_size), mu=0.0))

    preconditioner_fn = update_preconditioner(
        base_preconditioner, identity_vectorized, use_rank_1_update=True
    )

    def stopping_criterion(x: Array) -> Array:
        x_mat = to_matrix(x, hilbert_size)
        x_mat = 0.5 * (x_mat.conj().mT + x_mat)
        x_mat = x_mat / jnp.trace(x_mat)
        lindbladian_x = lindbladian(from_matrix(x_mat))
        if norm_type == 'max':
            norm = jnp.max(jnp.abs(lindbladian_x))
        else:
            norm = jnp.linalg.norm(lindbladian_x)
        return norm < tol

    krylov_size = min(krylov_size, hilbert_size - 1)

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

    state = finalize_density_matrix(to_matrix(x, n=hilbert_size), exact_dm)
    state = to_dm(from_matrix(state), n=hilbert_size, dims=hilbert_dimensions)
    return state, (n_iteration, success, U, C)
