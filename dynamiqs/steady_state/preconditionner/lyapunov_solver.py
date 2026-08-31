import equinox as eqx

import jax
import jax.numpy as jnp
from jax import Array


# IMPORTANT Note : The eigen decomposition should be performed in double precision
#                  (complex128) to ensure numerical stability.
#                  The Eig method provides significalntly better performances over
#                  the Bartel-Stewart method in order to solve the Lyapunov equation.
#                  However, the eigendecomposition can be unstable so it needs to be in
#                  double precision.
#                  The rest of the algorithm can be performed in single precision
#                  (complex64) without loss of accuracy, but the eigendecomposition step
#                  is critical and must be in double precision to ensure the overall
#                  stability and accuracy of the solver.


def _ensure_complex128(x: Array) -> Array:
    """Promote an array to complex128 (double precision).

    If the input is already ``complex128``, this is a no-op (JAX returns the
    same array without a copy).  Real ``float64`` inputs are promoted to
    ``complex128``; lower-precision dtypes (``float32``, ``complex64``, …)
    are up-cast accordingly.
    """
    return jnp.asarray(x, dtype=jnp.complex128)


class LyapunovSolverEig(eqx.Module):
    r"""Closed-form solver for continuous Lyapunov equations using eigendecomposition.

    This class provides an efficient solver for matrix equations of the form
    $$
        \mathcal{S}(X) = Y,
        \qquad
        \mathcal{S}(X) \coloneqq G X + X G^\dagger,
    $$
    where $G \in \mathbb{C}^{n\times n}$ is a fixed matrix and $\mu \in \mathbb{R}$
    is an optional shift parameter.

    The solution is obtained by diagonalizing $G$,
    $$
        G = U \Lambda U^{-1},
    $$
    and solving the Lyapunov equation elementwise in the eigenbasis:
    $$
        X = U \left(
            \frac{\widetilde{Y}_{ij}}
                 {\lambda_i + \bar{\lambda}_j + \mu}
        \right) U^\dagger,
        \qquad
        \widetilde{Y} = (U^{-1})^\dagger Y U^{-1}.
    $$
    This approach has $\mathcal{O}(n^3)$ preprocessing cost due to the
    eigendecomposition, but allows repeated solves with different right-hand
    sides $Y$ at low additional cost.

    Optionally, iterative refinement can be applied to improve the numerical
    accuracy of the solution. At each refinement step, the residual
    $R = Y - \mathcal{S}(X)$ is computed and a correction
    $\delta X = \mathcal{S}^{-1}(R)$ is added to $X$. This is particularly
    useful when $G$ is ill-conditioned or nearly defective, as the
    eigendecomposition-based solve may lose precision. Each refinement step
    costs $\mathcal{O}(n^3)$ (dominated by matrix multiplications).

    Args:
        G: Square matrix of shape `(n, n)` defining the Lyapunov operator.
            Automatically promoted to ``complex128`` (double precision) if
            not already; this is a no-op when the input is already
            ``complex128`` or ``float64``.
        n_refinement: Number of iterative refinement steps to apply after the
            initial eigendecomposition-based solve. Defaults to 0 (no
            refinement). A value of 1 or 2 is usually sufficient to recover
            full machine precision.

    Attributes:
        G: The matrix defining the Lyapunov operator.
        G_eigvals: Eigenvalues of `G`.
        G_eigvecs: Right eigenvectors of `G`.
        G_eigvecs_inv: Inverse (Hermitian-transposed) eigenvector matrix
            used for basis transformations.
        n_refinement: Number of iterative refinement steps.
    """

    G: Array
    G_eigvals: Array
    G_eigvecs: Array
    G_eigvecs_inv: Array
    n_refinement: int = 0

    def __init__(self, G: Array, n_refinement: int = 0):
        G = _ensure_complex128(G)
        self.G = G
        self.n_refinement = n_refinement

        self.G_eigvals, self.G_eigvecs = jnp.linalg.eig(self.G)
        self.G_eigvecs_inv = jnp.linalg.inv(self.G_eigvecs).mT.conj()

    def lyapunov(self, X: Array, mu: float) -> Array:
        """Apply the Lyapunov operator to a matrix X.

        Computes the right-hand side of the Lyapunov equation:
            G X + X G.H + mu X

        Args:
            X: Matrix of shape (n, n) to apply the Lyapunov operator to.
            mu: Shift parameter for the Lyapunov equation; may be useful for
                numerical stability.

        Returns:
            Result of applying the Lyapunov operator, shape (n, n).
            Equal to `G @ X + X @ G.H + mu * X.`
        """
        G = self.G
        return G @ X + X @ G.conj().mT + mu * X

    def lyapunov_adjoint(self, X: Array, mu: float) -> Array:
        """Apply the adjoint Lyapunov operator to a matrix X.

        Computes the right-hand side of the adjoint Lyapunov equation:
            G.H X + X G + mu X

        Args:
            X: Matrix of shape (n, n) to apply the Lyapunov operator to.
            mu: Shift parameter for the Lyapunov equation.

        Returns:
            Result of applying the adjoint Lyapunov operator, shape (n, n).
            Equal to `G.conj().T @ X + X @ G + mu * X.`
        """
        G = self.G
        return G.conj().T @ X + X @ G + mu * X

    def lyapunov_transpose(self, X: Array, mu: float) -> Array:
        """Apply the _transpose_ Lyapunov operator to a matrix X.

        Computes the right-hand side of the transpose Lyapunov equation:
            G.T X + X G.conj() + mu X

        Args:
            X: Matrix of shape (n, n) to apply the Lyapunov operator to.
            mu: Shift parameter for the Lyapunov equation.

        Returns:
            Result of applying the _transpose_ Lyapunov operator, shape (n, n).
            Equal to `G.T @ X + X @ G.conj() + mu * X.`

        Notes:
            Implemented for backward-mode automatic differentiation.
            Refer to `lyapu_adjoint` for the adjoint computation.
        """
        G = self.G
        return G.T @ X + X @ G.conj() + mu * X

    def _solve(self, Y: Array, mu: float) -> Array:
        r"""Solve the Lyapunov equation G X + X G.H + mu X = Y.

        Computes the solution via eigendecomposition, then optionally applies
        ``n_refinement`` iterative refinement steps. Each refinement step
        computes the residual $R = Y - \mathcal{S}(X)$ and corrects
        $X \leftarrow X + \mathcal{S}^{-1}(R)$ using the same
        eigendecomposition-based core solver.

        Args:
            Y: Right-hand side matrix of shape `(n, n)`.
            mu: Shift parameter.

        Returns:
            Solution matrix `X` of shape `(n, n)`.
        """
        Y = _ensure_complex128(Y)
        u_, v_, w_ = (self.G_eigvecs, self.G_eigvecs_inv, self.G_eigvals)

        def _core(Y: Array) -> Array:
            Y_tilde = v_.mT.conj() @ Y @ v_
            X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
            return u_ @ X_tilde @ u_.mT.conj()

        X = _core(Y)
        for _ in range(self.n_refinement):
            R = Y - self.lyapunov(X, mu)
            X = X + _core(R)
        return X

    def solve(self, Y: Array, mu: float) -> Array:
        return jax.lax.custom_linear_solve(
            lambda X: self.lyapunov(X, mu),
            Y,
            # `_mv` is the linear operator (matvec) passed by `custom_linear_solve`.
            # It is required by the API but unused here because we provide a closed-form
            # solver based on the eigendecomposition of G.
            # `_mvT` is the transpose matvec passed by `custom_linear_solve`.
            # It is required for autodiff, but unused since the transpose system
            # is solved analytically via a dedicated eigenspace-based routine.
            solve=lambda _mv, Y: self._solve(Y, mu),
            transpose_solve=lambda _mvT, Y: self._solve_transpose(Y, mu),
        )

    def solve_transpose(self, Y: Array, mu: float) -> Array:
        return jax.lax.custom_linear_solve(
            lambda X: self.lyapunov_transpose(X, mu),
            Y,
            solve=lambda _mv, Y: self._solve_transpose(Y, mu),
            transpose_solve=lambda _mvT, Y: self._solve(Y, mu),
        )

    def _solve_transpose(self, Y: Array, mu: float) -> Array:
        r"""Solve the _transpose_ equation $G^T X + X G^* + \mu X = Y$.

        Given $G = U \Lambda U^{-1}$, the direct solver uses the pair
        $(\tilde{U}, \tilde{V}) = (U,\; (U^{-1})^*)$ to change basis.
        For the transpose problem, note that
        $G^T = (U^{-1})^T \Lambda\, U^T = ((U^{-1})^*)^{-\dagger}\, \Lambda\,
        ((U^{-1})^*)^{\dagger}$,
        so the same ``_core`` routine applies with the substitution
        $$
            \tilde{U} \leftarrow (U^{-1})^*,
            \qquad
            \tilde{V} \leftarrow U^*.
        $$
        The eigenvalues $\Lambda$ are unchanged. Iterative refinement
        (``n_refinement`` steps) is applied identically to :meth:`_solve`.

        Args:
            Y: Right-hand side matrix of shape `(n, n)`.
            mu: Shift parameter.

        Returns:
            Solution matrix `X` of shape `(n, n)`.
        """
        Y = _ensure_complex128(Y)
        u_, v_, w_ = (self.G_eigvecs_inv.conj(), self.G_eigvecs.conj(), self.G_eigvals)

        def _core(Y: Array) -> Array:
            Y_tilde = v_.mT.conj() @ Y @ v_
            X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
            return u_ @ X_tilde @ u_.mT.conj()

        X = _core(Y)
        for _ in range(self.n_refinement):
            R = Y - self.lyapunov_transpose(X, mu)
            X = X + _core(R)
        return X
