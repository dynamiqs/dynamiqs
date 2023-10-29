from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor
from torch.linalg import cholesky_ex as cholesky

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils import cache, inv_sqrtm, kraus_map
from ..utils.utils import unit
from .me_solver import MESolver


class MERouchon(MESolver, AdjointFixedSolver):
    pass


class MERouchon1(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # forward time operators
        # M0: (b_H, 1, n, n)
        # M1s: (1, len(L), n, n)
        self._M0 = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)
        self._M1s = sqrt(self.dt) * self.L
        # reverse time operators
        # M0rev: (b_H, 1, n, n)
        self._M0rev = cache(lambda Hnh: self.I + 1j * self.dt * Hnh)

        if self.H.is_constant and self.options.cholesky_normalization:
            self.init_constant_cholesky()
        elif self.H.is_constant and self.options.sqrt_normalization:
            self.init_constant_sqrt()
        elif not self.H.is_constant and self.options.cholesky_normalization:
            self.init_time_dependent_cholesky()
        elif not self.H.is_constant and self.options.sqrt_normalization:
            raise ValueError(
                'Square root normalization of Rouchon solvers is not implemented for'
                ' time-dependent problems.'
            )
        else:
            self.init_no_normalization()

    def init_constant_cholesky(self):
        H = self.H(0.0)
        Hnh = self.Hnh(H)
        M0 = self._M0(Hnh)
        M0rev = self._M0rev(Hnh)
        T = cholesky(M0.mH @ M0 + self.dt * self.sum_LdagL)[0]
        Trev = cholesky(M0rev.mH @ M0rev - self.dt * self.sum_LdagL)[0]

        self.M0 = cache(lambda Hnh: torch.linalg.solve(T.mH, M0, left=False))
        self.M1s = torch.linalg.solve(T.mH, self._M1s, left=False)
        self.M0rev = cache(lambda Hnh: torch.linalg.solve(Trev.mH, M0rev, left=False))

    def init_constant_sqrt(self):
        H = self.H(0.0)
        Hnh = self.Hnh(H)
        M0 = self._M0(Hnh)
        M0rev = self._M0rev(Hnh)
        S = inv_sqrtm(M0.mH @ M0 + self.dt * self.sum_LdagL)
        Srev = inv_sqrtm(M0rev.mH @ M0rev - self.dt * self.sum_LdagL)

        self.M0 = cache(lambda Hnh: M0 @ S)
        self.M1s = self._M1s @ S
        self.M0rev = cache(lambda Hnh: M0rev @ Srev)

    def init_time_dependent_cholesky(self):
        self.M0 = self._M0
        self.M1s = self._M1s
        self.M0rev = self._M0rev
        # Cholesky decomposition of the Kraus map normalization
        self.T = cache(lambda M0: cholesky(M0.mH @ M0 + self.dt * self.sum_LdagL)[0])
        self.Trev = cache(
            lambda M0rev: cholesky(M0rev.mH @ M0rev - self.dt * self.sum_LdagL)[0]
        )

    def init_no_normalization(self):
        self.M0 = self._M0
        self.M1s = self._M1s
        self.M0rev = self._M0rev

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
        # H, Hnh, M0, T: (b_H, 1, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)

        # compute rho(t+dt)
        if self.options.cholesky_normalization and not self.H.is_constant:
            # renormalize the Kraus map: rho -> {T^{\dag}}^{-1} \rho T^{-1}
            T = self.T(M0)
            rho = torch.linalg.solve_triangular(T.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(T, rho, upper=False, left=False)

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

        # compute rho(t-dt)
        if self.options.cholesky_normalization and not self.H.is_constant:
            Trev = self.Trev(M0rev)
            rho = torch.linalg.solve_triangular(Trev.mH, rho, upper=True)
            rho = torch.linalg.solve_triangular(Trev, rho, upper=False, left=False)

        rho = kraus_map(rho, M0rev) - kraus_map(rho, self.M1s)

        # compute phi(t-dt)
        phi = kraus_map(phi, M0.mH) + kraus_map(phi, self.M1s.mH)

        if self.options.cholesky_normalization and not self.H.is_constant:
            T = self.T(M0)
            phi = torch.linalg.solve_triangular(T, phi, upper=False)
            phi = torch.linalg.solve_triangular(T.mH, phi, upper=True, left=False)

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
