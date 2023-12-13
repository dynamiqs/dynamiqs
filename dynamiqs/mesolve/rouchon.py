from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.linalg import cholesky_ex as cholesky

from .._utils import cache
from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils import inv_sqrtm
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
    def R(self, M0: Tensor, fwd: bool = True) -> Tensor:
        # `R` is close to identity but not exactly, we inverse it to normalize the
        # Kraus map in order to have a trace-preserving scheme
        # -> (..., n, n)
        dt = self.dt if fwd else -self.dt
        return M0.mH @ M0 + dt * self.sum_LdagL

    @cache(maxsize=2)
    def Ms(self, Hnh: Tensor, fwd: bool = True) -> tuple(Tensor, Tensor):
        # Kraus operators
        # -> (..., n, n), (nL, ..., n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh  # (..., n, n)
        M1s = sqrt(abs(dt)) * self.L  # (nL, ..., n, n)

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
        # -> (..., n, n)
        return cholesky(R)[0]  # lower triangular

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)
        H = self.H(t)  # (..., n, n)
        Hnh = self.Hnh(H)  # (..., n, n)
        M0, M1s = self.Ms(Hnh)  # (..., n, n), (nL, ..., n, n)

        # normalize the Kraus Map
        if self.options.normalize == 'cholesky':
            R = self.R(M0)  # (..., n, n)
            T = self.T(R)  # (..., n, n)
            rho = inv_kraus_matmul(T.mH, rho, upper=True)  # T.mH^-1 @ rho @ T^-1

        # compute rho(t+dt)
        rho = M0 @ rho @ M0.mH + (M1s @ rho @ M1s.mH).sum(0)  # (..., n, n)

        return unit(rho)

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        # rho: (..., n, n) -> (..., n, n)
        # phi: (..., n, n) -> (..., n, n)

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
        rho = M0rev @ rho @ M0rev.mH - (M1srev @ rho @ M1srev.mH).sum(0)

        # === forward time
        M0, M1s = self.Ms(Hnh)

        # compute phi(t-dt)
        phi = M0.mH @ phi @ M0 + (M1s.mH @ phi @ M1s).sum(0)

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
        # -> (..., n, n), (nL, ..., n, n)
        dt = self.dt if fwd else -self.dt
        M0 = self.I - 1j * dt * Hnh - 0.5 * dt**2 * Hnh @ Hnh
        M1s = 0.5 * sqrt(abs(dt)) * (self.L @ M0 + M0 @ self.L)

        return M0, M1s

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)

        # Note: for fast time-varying Hamiltonians, this method is not order 2 because
        # the  second-order time derivative term is neglected. This term could be added
        # in the zero-th order Kraus operator if needed, as
        # `M0 += -0.5j * dt**2 * \dot{H}`.

        H = self.H(t)  # (..., n, n)
        Hnh = self.Hnh(H)  # (..., n, n)
        M0, M1s = self.Ms(Hnh)  # (..., n, n), (nL, ..., n, n)

        # compute rho(t+dt)
        tmp = (M1s @ rho @ M1s.mH).sum(0)  # (..., n, n)
        rho = M0 @ rho @ M0.mH + tmp + 0.5 * (M1s @ tmp @ M1s.mH).sum(0)
        rho = unit(rho)  # (..., n, n)

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        # rho: (..., n, n) -> (..., n, n)
        # phi: (..., n, n) -> (..., n, n)

        # Note: for fast time-varying Hamiltonians, this method is not order 2 because
        # the  second-order time derivative term is neglected. This term could be added
        # in the zero-th order Kraus operator if needed, as
        # `M0 += -0.5j * dt**2 * \dot{H}`.

        H = self.H(t)
        Hnh = self.Hnh(H)

        # === reverse time
        M0rev, M1srev = self.Ms(Hnh, fwd=False)

        # compute rho(t-dt)
        tmp = (M1srev @ rho @ M1srev.mH).sum(0)
        rho = M0rev @ rho @ M0rev.mH - tmp + 0.5 * (M1srev @ tmp @ M1srev.mH).sum(0)
        rho = unit(rho)

        # === forward time
        M0, M1s = self.Ms(Hnh)

        # compute phi(t-dt)
        tmp = (M1s.mH @ phi @ M1s).sum(0)
        phi = M0.mH @ phi @ M0 + tmp + 0.5 * (M1s.mH @ tmp @ M1s).sum(0)

        return rho, phi
