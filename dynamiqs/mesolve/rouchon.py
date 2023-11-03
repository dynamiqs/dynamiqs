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


class MERouchon(MESolver, AdjointFixedSolver):
    pass


class MERouchon1(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # forward time operators
        # M0: (b_H, 1, n, n)
        # M1s: (1, len(L), n, n)
        self.M0 = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)
        self.M1s = sqrt(self.dt) * self.L

        # reverse time operators
        # M0rev: (b_H, 1, n, n)
        # M1srev: (1, len(L), n, n)
        self.M0rev = cache(lambda Hnh: self.I + 1j * self.dt * Hnh)
        self.M1srev = self.M1s

        if self.options.normalize:
            # `R` is close to identity but not exactly, we inverse it to normalize the
            # Kraus map in order to have a trace-preserving scheme
            self.R = cache(lambda M0: M0.mH @ M0 + self.dt * self.sum_LdagL)
            self.Rrev = cache(lambda M0rev: M0rev.mH @ M0rev - self.dt * self.sum_LdagL)

            if self.H.is_constant:
                # if H is constant, we normalize the operators once by computing
                # `S = sqrt(R)^-1` and applying is to the `Ms` operators
                H = self.H(0.0)
                Hnh = self.Hnh(H)
                M0 = self.M0(Hnh)
                M0rev = self.M0rev(Hnh)

                S = inv_sqrtm(self.R(M0))
                Srev = inv_sqrtm(self.Rrev(M0rev))

                self.M0 = cache(lambda Hnh: M0 @ S)
                self.M1s = self.M1s @ S
                self.M0rev = cache(lambda Hnh: Srev @ M0rev)
                self.M1srev = Srev @ self.M1s
            else:
                # if `H` is time-dependent, we normalize the map at each time step by
                # inverting `R` using its Cholesky decomposition `R = T @ T.mT`
                self.T = cache(lambda R: cholesky(R)[0])  # lower triangular
                self.Trev = cache(lambda Rrev: cholesky(Rrev)[0])  # lower triangular

        self.cholesky_normalization = self.options.normalize and not self.H.is_constant

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
        # H, Hnh, M0: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)

        # normalize the Kraus Map
        if self.cholesky_normalization:
            # R, T: (b_H, 1, n, n)
            R = self.R(M0)
            T = self.T(R)
            rho = inv_kraus_matmul(T.mH, rho, upper=True)  # T.mH^-1 @ rho @ T^-1

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, self.M1s)  # (b_H, b_rho, n, n)

        return unit(rho)

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
        # H, Hnh, M0, M0rev: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)
        M0rev = self.M0rev(Hnh)

        # normalize the Kraus Map
        if self.cholesky_normalization:
            # Rrev, Trev: (b_H, 1, n, n)
            Rrev = self.Rrev(M0rev)
            Trev = self.Trev(Rrev)
            rho = inv_kraus_matmul(Trev.mH, rho, upper=True)  # Tr.mH^-1 @ rho @ Tr^-1

        # compute rho(t-dt)
        rho = kraus_map(rho, M0rev) - kraus_map(rho, self.M1srev)

        # compute phi(t-dt)
        phi = kraus_map(phi, M0.mH) + kraus_map(phi, self.M1s.mH)

        if self.cholesky_normalization:
            # R, T: (b_H, 1, n, n)
            R = self.R(M0)
            T = self.T(R)
            phi = inv_kraus_matmul(T, phi, upper=False)  # T^-1 @ phi @ T.mH^-1

        return unit(rho), phi


class MERouchon2(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define cached operators
        # self.M0, self.M0dag, self.M0rev: (b_H, 1, n, n)
        # self.M1s, self.M1sdag: (b_H, len(L), n, n)
        self.M0 = cache(
            lambda Hnh: self.I - 1j * self.dt * Hnh - 0.5 * self.dt**2 * Hnh @ Hnh
        )
        self.M0dag = cache(lambda M0: M0.mH)
        self.M0rev = cache(
            lambda Hnh: self.I + 1j * self.dt * Hnh - 0.5 * self.dt**2 * Hnh @ Hnh
        )
        self.M1s = cache(lambda M0: 0.5 * sqrt(self.dt) * (self.L @ M0 + M0 @ self.L))
        self.M1sdag = cache(lambda M1s: M1s.mH)

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
        tmp = kraus_map(rho, M1s)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)  # (b_H, b_rho, n, n)
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
        M0dag = self.M0dag(M0)
        M0rev = self.M0rev(Hnh)
        M1s = self.M1s(M0)
        M1sdag = self.M1sdag(M1s)

        # compute rho(t-dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0rev) - tmp + 0.5 * kraus_map(tmp, M1s)
        rho = unit(rho)

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1sdag)
        phi = kraus_map(phi, M0dag) + tmp + 0.5 * kraus_map(tmp, M1sdag)

        return rho, phi
