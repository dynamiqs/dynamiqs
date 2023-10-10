from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils import cache, kraus_map
from ..utils.utils import unit
from .me_solver import MESolver


def chol(A: Tensor) -> Tensor:
    return torch.linalg.cholesky_ex(A)[0]


class MERouchon(MESolver, AdjointFixedSolver):
    pass


class MERouchon1(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # forward time operators
        self.M0 = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)
        self.M1s = sqrt(self.dt) * self.L
        self.T = cache(lambda M0: chol(M0.mH @ M0 + self.dt * self.sum_LdagL))

        # reverse time operators
        self.M0rev = cache(lambda Hnh: self.I + 1j * self.dt * Hnh)
        self.Trev = cache(
            lambda M0rev: chol(M0rev.mH @ M0rev - self.dt * self.sum_LdagL)
        )

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        # H, Hnh, M0, S, T: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)

        # compute rho(t+dt)
        if self.options.cholesky_normalization:
            T = self.T(M0)
            rho = torch.linalg.solve_triangular(T.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(T, rho, upper=False, left=False)

        rho = kraus_map(rho, M0) + kraus_map(rho, self.M1s)  # (b_H, b_rho, n, n)

        if not self.options.cholesky_normalization:
            rho = unit(rho)

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors of
            shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        # H, Hnh, M0, M0dag, M0rev: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)
        M0rev = self.M0rev(Hnh)

        # compute rho(t-dt)
        if self.options.cholesky_normalization:
            Trev = self.Trev(M0rev)
            rho = torch.linalg.solve_triangular(Trev.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(Trev, rho, upper=False, left=False)

        rho = kraus_map(rho, M0rev) - kraus_map(rho, self.M1s)

        if not self.options.cholesky_normalization:
            rho = unit(rho)

        # compute phi(t-dt)
        phi = kraus_map(phi, M0.mH) + kraus_map(phi, self.M1s.mH)

        if self.options.cholesky_normalization:
            T = self.T(M0)
            phi = torch.linalg.solve_triangular(T, phi, upper=False)
            phi = torch.linalg.solve_triangular(T.mH, phi, upper=True, left=False)

        return rho, phi


class MERouchon2(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.M0, self.M0rev, self.S1, self.T, self.Trev: (b_H, 1, n, n)
        # self.M1s: (b_H, len(L), n, n)

        # forward time operators
        self.M0 = cache(
            lambda Hnh: self.I - 1j * self.dt * Hnh - 0.5 * self.dt**2 * Hnh @ Hnh
        )
        self.M1s = cache(lambda M0: 0.5 * sqrt(self.dt) * (self.L @ M0 + M0 @ self.L))
        self.S1 = cache(
            lambda M0: (self.M1s(M0).mH @ self.M1s(M0)).sum(dim=1, keepdim=True)
        )
        self.T = cache(
            lambda M0: chol(M0.mH @ M0 + self.S1(M0) + 0.5 * self.S1(M0) @ self.S1(M0))
        )

        # reverse time operators
        self.M0rev = cache(
            lambda Hnh: self.I + 1j * self.dt * Hnh - 0.5 * self.dt**2 * Hnh @ Hnh
        )
        self.Trev = cache(
            lambda M0rev: chol(
                M0rev.mH @ M0rev
                - self.S1(M0rev)
                + 0.5 * self.S1(M0rev) @ self.S1(M0rev)
            )
        )

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 2.

        Notes:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        # H, Hnh, M0: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)
        M1s = self.M1s(M0)  # (b_H, len(L), n, n)

        # compute rho(t+dt)
        if self.options.cholesky_normalization:
            T = self.T(M0)
            rho = torch.linalg.solve_triangular(T.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(T, rho, upper=False, left=False)

        tmp = kraus_map(rho, M1s)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)  # (b_H, b_rho, n, n)

        if not self.options.cholesky_normalization:
            rho = unit(rho)  # (b_H, b_rho, n, n)

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
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors
            of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        # H, Hnh, M0, M0dag, M0rev: (b_H, 1, n, n)
        # M1s, M1sdag: (b_H, len(L), n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)
        M0rev = self.M0rev(Hnh)
        M1s = self.M1s(M0)

        # compute rho(t-dt)
        if self.options.cholesky_normalization:
            Trev = self.Trev(M0rev)
            rho = torch.linalg.solve_triangular(Trev.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(Trev, rho, upper=False, left=False)

        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0rev) - tmp + 0.5 * kraus_map(tmp, M1s)

        if not self.options.cholesky_normalization:
            rho = unit(rho)

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1s.mH)
        phi = kraus_map(phi, M0.mH) + tmp + 0.5 * kraus_map(tmp, M1s.mH)

        if self.options.cholesky_normalization:
            T = self.T(M0)
            phi = torch.linalg.solve_triangular(T, phi, upper=False)
            phi = torch.linalg.solve_triangular(T.mH, phi, upper=True, left=False)

        return rho, phi
