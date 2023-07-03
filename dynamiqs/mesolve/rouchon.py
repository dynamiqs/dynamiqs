from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..solvers.utils.utils import cache, inv_sqrtm, kraus_map
from ..utils.utils import trace
from .me_solver import MESolver


class MERouchon(MESolver, AdjointFixedSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = self.H.size(-1)
        self.I = torch.eye(self.n, device=self.H.device, dtype=self.H.dtype)  # (n, n)


class MERouchon1(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define cached operators
        self.M0 = cache(lambda H_nh: self.I - 1j * self.dt * H_nh)  # (b_H, 1, n, n)
        self.M0dag = cache(lambda M0: M0.mH)  # (b_H, 1, n, n)
        self.M0rev = cache(lambda H_nh: self.I + 1j * self.dt * H_nh)  # (b_H, 1, n, n)

        self.M1s = sqrt(self.dt) * self.jump_ops  # (1, len(jump_ops), n, n)
        self.M1sdag = self.M1s.mH  # (1, len(jump_ops), n, n)

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
        H = self.H(t)  # (b_H, 1, n, n)
        H_nh = self.H_nh(H)  # (b_H, 1, n, n)
        M0 = self.M0(H_nh)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, self.M1s)  # (b_H, b_rho, n, n)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(self, t: float, rho: Tensor, phi: Tensor):
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            dt: Time step.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors of
            shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        H = self.H(t)  # (b_H, 1, n, n)
        H_nh = self.H_nh(H)  # (b_H, 1, n, n)
        M0 = self.M0(H_nh)  # (b_H, 1, n, n)
        M0dag = self.M0dag(M0)  # (b_H, 1, n, n)
        M0rev = self.M0rev(H_nh)  # (b_H, 1, n, n)

        # compute rho(t-dt)
        rho = kraus_map(rho, M0rev) - kraus_map(rho, self.M1s)
        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        phi = kraus_map(phi, M0dag) + kraus_map(phi, self.M1sdag)

        return rho, phi


class MERouchon1_5(MERouchon):
    # TODO implement caching
    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.5.

        Note:
            No need for trace renormalization since the scheme is trace-preserving
            by construction.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_no_jump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * self.dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(self.dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.mH @ M0 + self.dt * self.sum_no_jump  # (b_H, 1, n, n)
        # TODO Fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)  # (b_H, b_rho, n, n)

        return rho

    def backward_augmented(self, t: float, rho: Tensor, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define cached operators
        self.M0 = cache(
            lambda H_nh: self.I - 1j * self.dt * H_nh - 0.5 * self.dt**2 * H_nh @ H_nh
        )  # (b_H, 1, n, n)
        self.M0dag = cache(lambda M0: M0.mH)  # (b_H, 1, n, n)
        self.M0rev = cache(
            lambda H_nh: self.I + 1j * self.dt * H_nh - 0.5 * self.dt**2 * H_nh @ H_nh
        )  # (b_H, 1, n, n)
        self.M1s = cache(
            lambda M0: 0.5 * sqrt(self.dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)
        )  # (b_H, len(jump_ops), n, n)
        self.M1sdag = cache(lambda M1s: M1s.mH)  # (b_H, len(jump_ops), n, n)

    def forward(self, t: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 2.

        Note:
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
        H = self.H(t)  # (b_H, 1, n, n)
        H_nh = self.H_nh(H)  # (b_H, 1, n, n)
        M0 = self.M0(H_nh)  # (b_H, 1, n, n)
        M1s = self.M1s(M0)  # (b_H, len(jump_ops), n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)  # (b_H, b_rho, n, n)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real  # (b_H, b_rho, n, n)

        return rho

    def backward_augmented(self, t: float, rho: Tensor, phi: Tensor):
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 2.

        Note:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.

        Args:
            t: Time.
            dt: Time step.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.
            phi: Adjoint state matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix and adjoint state matrix at previous time step, as tensors
            of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute cached operators
        H = self.H(t)
        H_nh = self.H_nh(H)
        M0 = self.M0(H_nh)
        M0dag = self.M0dag(M0)
        M0rev = self.M0rev(H_nh)
        M1s = self.M1s(M0)
        M1sdag = self.M1sdag(M1s)

        # compute rho(t-dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0rev) - tmp + 0.5 * kraus_map(tmp, M1s)
        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1sdag)
        phi = kraus_map(phi, M0dag) + tmp + 0.5 * kraus_map(tmp, M1sdag)

        return rho, phi
