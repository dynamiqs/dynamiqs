from __future__ import annotations

from math import sqrt

from torch import Tensor

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils import (
    CallableTDTensor,
    ConstantTDTensor,
    cache,
    inv_sqrtm,
    kraus_map,
)
from ..utils.utils import unit
from .me_solver import MESolver


class MERouchon(MESolver, AdjointFixedSolver):
    pass


class MERouchon1(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.options.sqrt_normalization:
            # define cached operators
            # self.M0, self.M0rev: (b_H, 1, n, n)
            self.M0 = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)
            self.M0rev = cache(lambda Hnh: self.I + 1j * self.dt * Hnh)

            # define M1s and M1srev
            # self.M1s: (1, len(L), n, n)
            self.M1s = sqrt(self.dt) * self.L
            self.M1srev = self.M1s
        else:
            # compute the inverse square root renormalization matrix
            # (for time-dependent Hamiltonians, exclude the Hamiltonian from
            # normalization)
            # Hnh, M0, M0rev, self.S, self.Srev: (b_H, 1, n, n)
            if isinstance(self.H, ConstantTDTensor):
                Hnh_const = self.Hnh(0.0)
            elif isinstance(self.H, CallableTDTensor):
                Hnh_const = -0.5j * self.sum_LdagL

            M0 = self.I - 1j * self.dt * Hnh_const
            M0rev = self.I + 1j * self.dt * Hnh_const
            self.S = inv_sqrtm(M0.mH @ M0 + self.dt * self.sum_LdagL)
            self.Srev = inv_sqrtm(M0rev.mH @ M0rev - self.dt * self.sum_LdagL)

            # define cached operators
            # self.M0, self.M0rev: (b_H, 1, n, n)
            self.M0 = cache(lambda Hnh: self.S - 1j * self.dt * Hnh @ self.S)
            self.M0rev = cache(lambda Hnh: self.Srev + 1j * self.dt * Hnh @ self.Srev)

            # define M1s and M1srev
            # self.M1s, self.M1srev: (1, len(L), n, n)
            self.M1s = sqrt(self.dt) * self.L @ self.S
            self.M1srev = sqrt(self.dt) * self.L @ self.Srev

        self.M0dag = cache(lambda M0: M0.mH)  # (b_H, 1, n, n)
        self.M1sdag = self.M1s.mH  # (1, len(L), n, n)

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

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, self.M1s)  # (b_H, b_rho, n, n)
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
        # t is passed in as negative time

        # compute cached operators
        # H, Hnh, M0, M0dag, M0rev: (b_H, 1, n, n)
        H = self.H(-t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)
        M0dag = self.M0dag(M0)
        M0rev = self.M0rev(Hnh)

        # compute rho(t-dt)
        rho = kraus_map(rho, M0rev) - kraus_map(rho, self.M1srev)
        rho = unit(rho)

        # compute phi(t-dt)
        phi = kraus_map(phi, M0dag) + kraus_map(phi, self.M1sdag)

        return rho, phi


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
        # t is passed in as negative time

        # compute cached operators
        # H, Hnh, M0, M0dag, M0rev: (b_H, 1, n, n)
        # M1s, M1sdag: (b_H, len(L), n, n)
        H = self.H(-t)
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
