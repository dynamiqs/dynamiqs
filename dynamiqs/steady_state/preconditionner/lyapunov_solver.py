import functools as ft
import warnings
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class BaseLyapunovEquation(eqx.Module):
    G: Array

    def lyapunov(self, X: Array, mu: float):
        """Apply the Lyapunov operator to a matrix X.

        Computes the right-hand side of the Lyapunov equation:
            G X + X G.H + mu X

        Args:
            X: Matrix of shape (n, n) to apply the Lyapunov operator to.
            mu: Shift parameter for the Lyapunov equation.

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

    @abstractmethod
    def _solve(self, Y: Array, mu: float):
        pass

    @abstractmethod
    def _solve_adjoint(self, Y: Array, mu: float) -> Array:
        pass

    @abstractmethod
    def _solve_transpose(self, Y: Array, mu: float) -> Array:
        pass

    def solve(self, Y: Array, mu: float) -> Array:

        return jax.lax.custom_linear_solve(
            lambda X: self.lyapunov(X, mu),
            Y,
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


class LyapuSolverEig(BaseLyapunovEquation):
    G_eigvals: Array
    G_eigvecs: Array
    G_eigvecs_inv: Array

    def __init__(self, G: Array):
        # G = G.astype(jnp.complex64)
        self.G = G

        self.G_eigvals, self.G_eigvecs = jnp.linalg.eig(self.G)
        self.G_eigvecs_inv = jnp.linalg.inv(self.G_eigvecs).mT.conj()

    def _solve(self, Y: Array, mu: float):
        """Solve the Lyapunov equation G X + X G.H + mu X = Y."""
        u_, v_, w_ = (self.G_eigvecs, self.G_eigvecs_inv, self.G_eigvals)

        Y_tilde = v_.mT.conj() @ Y @ v_
        X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
        X = u_ @ X_tilde @ u_.mT.conj()
        return X

    def _solve_adjoint(self, Y: Array, mu: float) -> Array:
        """Solves the adjoint equation G.H X + X G + mu X = Y.

        Notes:
            Uses the flip trick to transform the adjoint problem.
            The transformation uses:
                Z_t = Z J
                T_t = J T^H J
        """
        u_, v_, w_ = (self.G_eigvecs_inv, self.G_eigvecs, self.G_eigvals.conj())

        Y_tilde = v_.mT.conj() @ Y @ v_
        X_tilde = Y_tilde / (w_[:, None] + w_[None, :].conj() + mu)
        X = u_ @ X_tilde @ u_.mT.conj()
        return X

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
        X = u_ @ X_tilde @ u_.mT.conj()
        return X
