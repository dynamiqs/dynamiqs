import abc
from collections.abc import Callable
from typing import Any

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..linear_system.gmres import gmres
from ..preconditionner.lyapunov_solver import LyapunovSolverEig
from .utils import finalize_density_matrix, frobenius_dot_product, update_preconditioner


class GMRESAuxInfo(eqx.Module):
    n_iteration: int
    success: Array | bool
    recycling: tuple[Array, Array]


def steady_state(
    H: dq.QArray,
    Ls: list[dq.QArray],
    tol: float = 1e-4,
    *,
    initial_guess: dq.QArray | None = None,
    max_iteration: int = 100,
    krylov_size: int = 32,
    recycling: int = 5,
    exact_dm: bool = True,
    norm_type: str = 'max',
) -> tuple[dq.QArray, GMRESAuxInfo]:
    r"""Compute the steady state of the Lindblad master equation.

    This function finds the density matrix $\rho_\mathrm{ss}$ satisfying
    $$
        \mathcal{L}(\rho_\mathrm{ss}) = -i[H, \rho_\mathrm{ss}]
        + \sum_{k=1}^N \left(
            L_k \rho_\mathrm{ss} L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho_\mathrm{ss}
            - \frac{1}{2} \rho_\mathrm{ss} L_k^\dag L_k
        \right) = 0,
    $$
    where $H$ is the system's Hamiltonian and $\{L_k\}$ is a collection of jump
    operators.

    The steady state is computed using a preconditioned GMRES algorithm. The
    Lindbladian is deflated with a rank-1 update to enforce the trace constraint,
    and the resulting linear system is preconditioned by a Lyapunov equation solver.
    The returned density matrix is Hermitian with unit trace.

    # Method

    ## Deflated linear system

    The Lindbladian superoperator $\mathcal{L}$ is singular (it has a zero
    eigenvalue corresponding to the steady state). To obtain a non-singular system,
    a rank-1 deflation is applied:
    $$
        \left(\mathcal{L} +  \,
        \mathrm{vec}(I)\,\mathrm{vec}(I)^\top\right)
        \mathrm{vec}(\rho) = \, \mathrm{vec}(I),
    $$
    $I$ is the identity matrix, and $\mathrm{vec}(\cdot)$ denotes
    column-major vectorization. This shift lifts the zero eigenvalue while
    preserving the steady-state solution.

    ## Lyapunov preconditioner

    The preconditioner is built from the *non-recycling* part of the Lindbladian,
    i.e. the Lyapunov superoperator $\mathcal{S}$ defined by
    $$
        \mathcal{S}(\rho) = G\rho + \rho G^\dag,
    $$
    where
    $$
        G = iH + \tfrac{1}{2}\sum_k L_k^\dag L_k.
    $$
    Note that $G$ captures both the Hamiltonian evolution and the decay terms, but
    omits the recycling (or "quantum jump") terms $L_k \rho L_k^\dag$. Inverting
    $\mathcal{S}$ amounts to solving a Lyapunov equation, which is done
    analytically via eigendecomposition of $G$: given the decomposition
    $G = U \Lambda U^{-1}$, the solution of $GX + XG^\dag + \mu X = Y$ is
    $$
        X = U \widetilde{X} U^\dag, \qquad
        \widetilde{X}_{ij} = \frac{\widetilde{Y}_{ij}}
        {\lambda_i + \bar\lambda_j + \mu},
    $$
    where $\widetilde{Y} = U^{-1}^\dag Y U^{-1}$.

    Args:
        H (qarray of shape (n, n)): Hamiltonian.
        Ls (list of qarray, each of shape (n, n)): List of jump operators.
        tol: Tolerance for the stopping criterion. The solver stops when
            $\|\mathcal{L}(\rho)\| < \mathrm{tol}$, where the norm is determined
            by `norm_type`. Defaults to `1e-4`. This tolerance works for both simple and double precision.
        initial_guess (qarray of shape (n, n), optional): Initial guess for the
            density matrix. Defaults to `None`, which uses the vacuum state
            $|0\rangle\langle 0|$.
        max_iter: Maximum number of outer GMRES iterations. Defaults to `100`.
        krylov_size: Size of the Krylov subspace used in each GMRES restart cycle.
            Defaults to `32`.
            If the steady state doesn't converge to the wanted tolerance, this parameter can be increased:
            `krylov_size = 64` or `128`
        exact_dm: If `True`, the final density matrix is projected onto the set of
            valid density matrices (positive semi-definite with unit trace) by
            diagonalizing and projecting the eigenvalues onto the simplex. If
            `False`, only Hermitization and trace normalization are applied.
            Defaults to `True`.
        norm_type: Norm used in the stopping criterion. Supported values:
            `'max'` (element-wise maximum of $|\mathcal{L}(\rho)|$) and
            `'norm2'` (Frobenius norm of $\mathcal{L}(\rho)$). Defaults to
            `'max'`.

    Returns:
        A tuple `(rho_ss, info)` where:

        - **`rho_ss`** _(qarray of shape (n, n))_ - The steady-state density matrix.
        - **`info`** _(`GMRESAuxInfo`)_ - Informations about the solve.

                **Attributes:**

                - **`n_iteration`** _(int)_ - Number of outer GMRES iterations
                    performed.
                - **`success`** _(array or bool)_ - Whether the solver converged
                    within the specified tolerance.
                - **`recycling`** _(tuple[array, array])_ - Recycled Krylov vectors
                    `(U, C)` that can be reused in subsequent solves.

    Examples:
    ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)

        H = a.dag() @ a
        jump_ops = [a]

        rho_ss, info = steady_state(H, jump_ops)
        print(f"Converged: {info.success}, iterations: {info.n_iteration}")
    ```
    """
    supported_norm_types = ('max', 'norm2')
    if norm_type not in supported_norm_types:
        raise ValueError(
            f'Unknown norm type {norm_type!r}. Expected one of {supported_norm_types}.'
        )

    n = H.shape[-1]
    dims = H.dims

    # Conversion of Ls in tensor in order to efficiently get LdagL
    Ls_q = dq.stack(Ls)
    LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()

    G = 1j * H.to_jax() + 1 / 2 * LdagL
    dtype = G.dtype

    # The Eigen decomposition called in the solver is not differentiable
    # On the other hand, we're differentiating the result of solve with
    # implicit differentiation: we don't need to differentiate through
    # the solver itself. Hence the stop_gradient here.
    preconditioner = LyapunovSolverEig(jax.lax.stop_gradient(G))

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

    identity_vectorized = from_matrix(jnp.eye(n, dtype=dtype))
    rhs = identity_vectorized

    def lindbladian(x: Array) -> Array:
        return from_dm(dq.lindbladian(H, Ls_q, to_dm(x)))

    def lindbladian_plus_rank1(x: Array) -> Array:
        return (
            lindbladian(x)
            # dot do not perform any conjugation,
            # which is fine since identity is real
            + identity_vectorized.dot(x) * identity_vectorized
        )

    def base_preconditioner(x: Array) -> Array:
        return -from_matrix(preconditioner.solve(to_matrix(x), mu=0.0))

    # When `use_rank_1_update` is `True`, the preconditioner is further corrected
    # to account for the rank-1 deflation using the Sherman-Morrison formula.
    use_rank_1_update = True
    preconditioner_fn = update_preconditioner(
        base_preconditioner, identity_vectorized, use_rank_1_update
    )

    def stopping_criterion(x: Array) -> Array:
        """Checks if the hermicized, trace-1 density matrix satisfies
        `|| L rho || < tol`."""
        x_mat = to_matrix(x)
        x_mat = 0.5 * (x_mat.conj().mT + x_mat)
        x_mat = x_mat / jnp.trace(x_mat)
        if norm_type == 'max':
            norm = jnp.max(jnp.abs(lindbladian(from_matrix(x_mat))))
        else:  # norm_type == 'norm2' (validated above)
            norm = jnp.linalg.norm(lindbladian(from_matrix(x_mat)))
        return norm < tol

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
    recycling_info = (U, C)

    rho = finalize_density_matrix(to_matrix(x), exact_dm)
    rho = to_dm(from_matrix(rho))

    return rho, GMRESAuxInfo(n_iteration, success, recycling_info)
