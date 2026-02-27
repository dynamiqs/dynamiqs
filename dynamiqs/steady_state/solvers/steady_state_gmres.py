from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse

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
from ..preconditionner.lyapunov_solver import LyapunovSolverEig


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
    r"""GMRES steady-state solver with Krylov recycling.

    Solves the deflated linear system
    $$
        (\mathcal{L} + |I\rangle\langle I|) |x\rangle = |I\rangle
    $$
    via preconditioned GMRES. Differentiation is handled by
    `jax.lax.custom_linear_solve`.

    GMRES steady-state solver with Krylov recycling.

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

        print(result.rho)
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
        del options

        n = H.shape[-1]
        dims = H.dims
        tol = self.tol
        max_iter = self.max_iteration
        krylov_size = min(self.krylov_size, n * n - 1)

        H_jax = H.to_jax()
        Ls_jax = [L.to_jax() for L in Ls]
        dtype = H_jax.dtype

        identity_vec = from_matrix(jnp.eye(n, dtype=dtype))

        if rho0 is None:
            rho0 = dq.coherent_dm(n, 0.0)
        x_0 = from_dm(rho0)

        H_q = dq.asqarray(H_jax, dims=dims)
        Ls_q = dq.stack(
            [dq.asqarray(L_j, dims=Ls[i].dims) for i, L_j in enumerate(Ls_jax)]
        )

        def lindbladian_vec(x):
            return from_dm(dq.lindbladian(H_q, Ls_q, to_dm(x, n=n, dims=dims)))

        def deflated_matvec(x):
            return lindbladian_vec(x) + identity_vec * jnp.dot(identity_vec, x)

        # ── preconditioners (stop_gradient: not differentiated) ─────────────
        LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()
        G = jax.lax.stop_gradient(1j * H_jax + 0.5 * LdagL)

        def make_preconditioner(G_mat):
            solver = LyapunovSolverEig(G_mat)

            def precond(x):
                return -from_matrix(solver.solve(to_matrix(x, n=n), mu=0.0))

            return update_preconditioner(precond, identity_vec, use_rank_1_update=True)

        precond_fn = make_preconditioner(G)
        precond_fn_adj = make_preconditioner(G.conj().T)

        # ── custom_linear_solve ─────────────────────────────────────────────

        def solve(matvec, b):
            x, _info = jax_sparse.gmres(
                matvec,
                b,
                x0=x_0,
                tol=tol,
                restart=krylov_size,
                maxiter=max_iter,
                M=precond_fn,
            )
            return x

        def transpose_solve(matvec_adj, b):
            y, _info = jax_sparse.gmres(
                matvec_adj,
                b,
                x0=jnp.zeros_like(b),
                tol=tol,
                restart=krylov_size,
                maxiter=max_iter,
                M=precond_fn_adj,
            )
            return y

        x_sol = jax.lax.custom_linear_solve(
            deflated_matvec,
            identity_vec,
            solve=solve,
            transpose_solve=transpose_solve,
            symmetric=False,
        )

        rho_ss = finalize_density_matrix(to_matrix(x_sol, n=n), self.exact_dm)

        state_q = dq.asqarray(rho_ss, dims=dims)
        return SteadyStateGMRESResult(rho=state_q)
