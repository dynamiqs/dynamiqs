from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.linalg import cholesky_ex as cholesky

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils import cache, inv_sqrtm, kraus_map
from ..utils.utils import unit
from .me_solver import MESolver


def inv_kraus_matmul(A: Tensor, B: Tensor, upper: bool) -> Tensor:
    # -> A^-1 @ B @ A.mH^-1
    B = torch.linalg.solve_triangular(A, B, upper=upper)
    B = torch.linalg.solve_triangular(A.mH, B, upper=not upper, left=False)
    return B


def batched_mult_left(x, y):
    r"""Computes the batched matrix product of two tensors where the left tensor
    has one more dimension than the right one.
     Args:
         x: Tensor of shape (N, ..., n, k)
         y: Tensor of shape (..., k, m)

     Returns:
        The batched product of `x @ y`
         `(N, ..., n, m)`.
    """
    return torch.einsum("n...ik,...kj->n...ij", x, y)


def batched_mult_right(x, y):
    r"""Computes the batched matrix product of two tensors where the right tensor
    has one more dimension than the left one.
     Args:
         x: Tensor of shape (N, ..., n, k)
         y: Tensor of shape (..., k, m)

     Returns:
        The batched product of `x @ y`
         `(N, ..., n, m)`.
    """
    return torch.einsum("...ik,n...kj->n...ij", x, y)


class MERouchon(MESolver, AdjointFixedSolver):
    pass


class MERouchon1(MERouchon):
    @cache(maxsize=2)
    def R(self, M0: Tensor, fwd: bool = True) -> Tensor:
        # `R` is close to identity but not exactly, we inverse it to normalize the
        # Kraus map in order to have a trace-preserving scheme
        # -> (b_H, b_L, 1, n, n)
        dt = self.dt if fwd else -self.dt
        return M0.mH @ M0 + dt * self.sum_LdagL

    @cache(maxsize=2)
    def Ms(self, Hnh: Tensor, fwd: bool = True) -> tuple(Tensor, Tensor):
        # Kraus operators
        # -> (b_H, b_L, 1, n, n), (len(L), 1, b_L, 1, n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh  # (b_H, b_L, 1, n, n)
        M1s = sqrt(abs(dt)) * self.L  # (len(L), 1, b_L, 1, n, n)

        if self.options.normalize == 'sqrt':
            R = self.R(M0, fwd=fwd)
            # we normalize the operators by computing `S = sqrt(R)^-1` and applying it
            # to the `Ms` operators
            S = inv_sqrtm(R)
            M0 = M0 @ S if fwd else S @ M0
            M1s = M1s @ S if fwd else S @ M1s

        return M0, M1s

    @cache(maxsize=2)
    def T(self, R: Tensor) -> Tensor:
        # we normalize the map at each time step by inverting `R` using its Cholesky
        # decomposition `R = T @ T.mT`
        # -> (b_H, b_L, 1, n, n)
        return cholesky(R)[0]  # lower triangular

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_L, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape
            `(b_H, b_L, b_rho, n, n)`.
        """
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        H = self.H(t)  # (b_H, 1, 1, n, n)
        Hnh = self.Hnh(H)  # (b_H, b_L, 1, n, n)
        M0, M1s = self.Ms(Hnh)  # (b_H, b_L, 1, n, n), (len(L), 1, b_L, 1, n, n)

        # normalize the Kraus Map
        if self.options.normalize == 'cholesky':
            R = self.R(M0)  # (b_H, b_L, 1, n, n)
            T = self.T(R)  # (b_H, b_L, 1, n, n)
            rho = inv_kraus_matmul(T.mH, rho, upper=True)  # T.mH^-1 @ rho @ T^-1

        # compute rho(t+dt)
        rho = M0 @ rho @ M0.mH + kraus_map(rho, M1s)  # (b_H, b_L, b_rho, n, n)

        return unit(rho)

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 1.

        Args:
            t: Time (negative-valued).
            rho: Density matrix of shape `(b_H, b_L, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_L, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors of
            shape `(b_H, b_L, b_rho, n, n)`.
        """
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        # phi: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)

        H = self.H(t)
        Hnh = self.Hnh(H)

        # === reverse time
        M0rev, M1srev = self.Ms(Hnh, fwd=False)

        # normalize the Kraus Map
        if self.options.normalize == 'cholesky':
            Rrev = self.R(M0rev, fwd=False)
            Trev = self.T(Rrev)
            rho = inv_kraus_matmul(Trev.mH, rho, upper=True)  # Tr.mH^-1 @ rho @ Tr^-1

        # compute rho(t-dt)
        rho = M0rev @ rho @ M0rev.mH - kraus_map(rho, M1srev)

        # === forward time
        M0, M1s = self.Ms(Hnh)

        # compute phi(t-dt)
        phi = M0.mH @ phi @ M0 + kraus_map(phi, M1s.mH)

        # normalize the Kraus Map
        if self.options.normalize == 'cholesky':
            R = self.R(M0)
            T = self.T(R)
            phi = inv_kraus_matmul(T, phi, upper=False)  # T^-1 @ phi @ T.mH^-1

        return unit(rho), phi


class MERouchon2(MERouchon):
    @cache(maxsize=2)
    def Ms(self, Hnh: Tensor, fwd: bool = True) -> tuple(Tensor, Tensor):
        # Kraus operators
        # -> (b_H, b_L, 1, n, n), (len(L), b_H, b_L, 1, n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh - 0.5 * dt**2 * Hnh @ Hnh  # (b_H, b_L, 1, n, n)

        M1s = (
            0.5
            * sqrt(abs(dt))
            * (batched_mult_left(self.L, M0) + batched_mult_right(M0, self.L))
        )  # (len(L), b_H, b_L, 1, n, n)

        return M0, M1s

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 2.

        Notes:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_L, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape
            `(b_H, b_L, b_rho, n, n)`.
        """
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)

        H = self.H(t)  # (b_H, 1, 1, n, n)
        Hnh = self.Hnh(H)  # (b_H, 1, 1, n, n)
        M0, M1s = self.Ms(Hnh)  # (b_H, b_L, 1, n, n), (len(L), b_H, b_L, 1, n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)  # (b_H, b_L, b_rho, n, n)
        rho = (
            M0 @ rho @ M0.mH + tmp + 0.5 * kraus_map(tmp, M1s)
        )  # (b_H, b_L, b_rho, n, n)
        rho = unit(rho)  # (b_H, b_L, b_rho, n, n)

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 2.

        Notes:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.

        Args:
            t: Time (negative-valued).
            rho: Density matrix of shape `(b_H, b_L, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_L, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors
            of shape `(b_H, b_L, b_rho, n, n)`.
        """
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        # phi: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)

        H = self.H(t)
        Hnh = self.Hnh(H)

        # === reverse time
        M0rev, M1srev = self.Ms(Hnh, fwd=False)

        # compute rho(t-dt)
        tmp = kraus_map(rho, M1srev)
        rho = M0rev @ rho @ M0rev.mH - tmp + 0.5 * kraus_map(tmp, M1srev)
        rho = unit(rho)

        # === forward time
        M0, M1s = self.Ms(Hnh)

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1s.mH)
        phi = M0.mH @ phi @ M0 + tmp + 0.5 * kraus_map(tmp, M1s.mH)

        return rho, phi
