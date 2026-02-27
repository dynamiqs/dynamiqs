import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class LyapunovSolverEig(eqx.Module):
    r"""Closed-form solver for continuous Lyapunov equations using eigendecomposition.

    This class provides an efficient solver for matrix equations of the form
    $$
        G X + X G^\dagger + \mu X = Y,
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

    Args:
        G: Square matrix of shape `(n, n)` defining the Lyapunov operator.

    Attributes:
        G: The matrix defining the Lyapunov operator.
        G_eigvals: Eigenvalues of `G`.
        G_eigvecs: Right eigenvectors of `G`.
        G_eigvecs_inv: Inverse (Hermitian-transposed) eigenvector matrix
            used for basis transformations.
    """

    G: Array
    G_eigvals: Array
    G_eigvecs: Array
    G_eigvecs_inv: Array

    def __init__(self, G: Array):
        self.G = G

        self.G_eigvals, self.G_eigvecs = jnp.linalg.eig(self.G)
        self.G_eigvecs_inv = jnp.linalg.inv(self.G_eigvecs).mT.conj()

    def lyapunov(self, X: Array, mu: float) -> Array:
        """Apply the Lyapunov operator to a matrix X.

        Computes the right-hand side of the Lyapunov equation:
            G X + X G.H + mu X

        Args:
            X: Matrix of shape (n, n) to apply the Lyapunov operator to.
            mu: Shift parameter for the Lyapunov equation,
            may be useful for numerical stability.

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
        """Solve the Lyapunov equation G X + X G.H + mu X = Y."""
        u_, v_, w_ = (self.G_eigvecs, self.G_eigvecs_inv, self.G_eigvals)

        Y_tilde = v_.mT.conj() @ Y @ v_
        X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
        return u_ @ X_tilde @ u_.mT.conj()

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
        """Solves the _transpose_ equation G.T X + X G* + mu X = Y.

        Notes:
            Uses the flip trick to transform the transpose problem.
            The transformation uses:
                Z_t = Z.conj()[:, ::-1]
                T_t = T.T[::-1, ::-1]
        """
        u_, v_, w_ = (self.G_eigvecs_inv.conj(), self.G_eigvecs.conj(), self.G_eigvals)

        Y_tilde = v_.mT.conj() @ Y @ v_
        X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
        return u_ @ X_tilde @ u_.mT.conj()
