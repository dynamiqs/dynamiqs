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
    @cache(maxsize=2)
    def Ms(self, Hnh: Tensor, fwd: bool = True) -> tuple(Tensor, Tensor):
        # Kraus operators
        # -> (b_H, 1, n, n), (1, len(L), n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh  # (b_H, 1, n, n)
        M1s = sqrt(abs(dt)) * self.L  # (1, len(L), n, n)

        if self.options.normalize == 'sqrt':
            R = self.R(M0, fwd=fwd)
            # we normalize the operators by computing `S = sqrt(R)^-1` and applying it
            # to the `Ms` operators
            S = inv_sqrtm(R)
            M0 = M0 @ S if fwd else S @ M0
            M1s = M1s @ S if fwd else S @ M1s

        return M0, M1s

    @cache(maxsize=2)
    def R(self, M0: Tensor, fwd: bool = True) -> Tensor:
        # `R` is close to identity but not exactly, we inverse it to normalize the
        # Kraus map in order to have a trace-preserving scheme
        # -> (b_H, 1, n, n)
        dt = self.dt if fwd else -self.dt
        return M0.mH @ M0 + dt * self.sum_LdagL

    @cache(maxsize=2)
    def T(self, R: Tensor) -> Tensor:
        # we normalize the map at each time step by inverting `R` using its Cholesky
        # decomposition `R = T @ T.mT`
        # -> (b_H, 1, n, n)
        return cholesky(R)[0]  # lower triangular

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        H = self.H(t)  # (b_H, 1, n, n)
        Hnh = self.Hnh(H)  # (b_H, 1, n, n)
        M0, M1s = self.Ms(Hnh)  # (b_H, 1, n, n), (1, len(L), n, n)

        # normalize the Kraus Map
        if self.options.normalize == 'cholesky':
            R = self.R(M0)  # (b_H, 1, n, n)
            T = self.T(R)  # (b_H, 1, n, n)
            rho = inv_kraus_matmul(T.mH, rho, upper=True)  # T.mH^-1 @ rho @ T^-1

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, M1s)  # (b_H, b_rho, n, n)

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
        rho = kraus_map(rho, M0rev) - kraus_map(rho, M1srev)

        # === forward time
        M0, M1s = self.Ms(Hnh)

        # compute phi(t-dt)
        phi = kraus_map(phi, M0.mH) + kraus_map(phi, M1s.mH)

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
        # -> (b_H, 1, n, n), (b_H, len(L), n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh - 0.5 * dt**2 * Hnh @ Hnh  # (b_H, 1, n, n)
        M1s = 0.5 * sqrt(abs(dt)) * (self.L @ M0 + M0 @ self.L)  # (b_H, len(L), n, n)

        return M0, M1s

    @cache(maxsize=2)
    def Ms_S(self, M0: Tensor, M1s: Tensor, fwd: bool = True) -> tuple(Tensor, Tensor):
        if self.options.normalize == 'sqrt':
            R = self.R(M0, M1s, fwd=fwd)
            # we normalize the operators by computing `S = sqrt(R)^-1` and applying it
            # to the `Ms` operators
            S = inv_sqrtm(R)
            M0 = M0 @ S if fwd else S @ M0
            M1s = M1s @ S if fwd else S @ M1s

        return M0, M1s

    @cache(maxsize=2)
    def R(self, M0: Tensor, M1s: Tensor, fwd: bool = True) -> Tensor:
        # `R` is close to identity but not exactly, we inverse it to normalize the
        # Kraus map in order to have a trace-preserving scheme
        # -> (b_H, 1, n, n)
        sign_fwd = 1 if fwd else -1
        R0 = M0.mH @ M0
        R1 = (M1s.mH @ M1s).sum(dim=1, keepdim=True)
        R2 = self.R2

        return R0 + sign_fwd * R1 + R2

    @property
    def R2(self) -> Tensor:
        M2s = self.L.unsqueeze(1) @ self.L.unsqueeze(2)
        M2s = M2s.view(1, self.L.size(1) ** 2, *self.L.size()[2:])
        return 0.5 * self.dt**2 * (M2s.mH @ M2s).sum(dim=1, keepdim=True)

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

        H = self.H(t)  # (b_H, 1, n, n)
        Hnh = self.Hnh(H)  # (b_H, 1, n, n)
        M0, M1s = self.Ms(Hnh)  # (b_H, 1, n, n), (b_H, len(L), n, n)
        M0_S, M1s_S = self.Ms_S(M0, M1s)  # (b_H, 1, n, n), (b_H, len(L), n, n)

        # compute rho(t+dt)
        # tmp, rho: (b_H, b_rho, n, n)
        tmp = kraus_map(rho, M1s_S)
        rho = kraus_map(rho, M0_S) + tmp + 0.5 * kraus_map(tmp, M1s)
        rho = unit(rho)

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

        H = self.H(t)
        Hnh = self.Hnh(H)

        # === reverse time
        M0rev, M1srev = self.Ms(Hnh, fwd=False)
        S_M0rev, S_M1srev = self.Ms_S(M0rev, M1srev, fwd=False)

        # compute rho(t-dt)
        tmp = kraus_map(rho, S_M1srev)
        rho = kraus_map(rho, S_M0rev) - tmp + 0.5 * kraus_map(tmp, M1srev)
        rho = unit(rho)

        # === forward time
        M0, M1s = self.Ms(Hnh)
        M0_S, M1s_S = self.Ms_S(M0, M1s)

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1s_S.mH)
        phi = kraus_map(phi, M0_S.mH) + tmp + 0.5 * kraus_map(tmp, M1s.mH)

        return rho, phi
