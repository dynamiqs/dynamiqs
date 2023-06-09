from __future__ import annotations

from abc import abstractmethod
from math import sqrt

import torch
from methodtools import lru_cache
from torch import Tensor

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from ..utils.solver_utils import inv_sqrtm, kraus_map
from ..utils.utils import trace
from .me_solver import MESolver


class MERouchon(MESolver, AdjointFixedSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = self.H.size(-1)
        self.I = torch.eye(self.n, device=self.H.device, dtype=self.H.dtype)  # (n, n)
        self.dt = self.options.dt

    @abstractmethod
    def M0(self, H_nh: Tensor, dt: float) -> Tensor:
        """Compute the zero-th order Kraus operator at time $t$.

        Args:
            t: Time.
            dt: Time step.

        Returns:
            Zero-th order Kraus operator of shape `(b_H, 1, n, n)`.
        """
        # -> (b_H, 1, n, n)
        pass

    @lru_cache(maxsize=1)
    def M0dag(self, M0: Tensor) -> Tensor:
        """Compute the adjoint of the zero-th order Kraus operator at time $t$."""
        # -> (b_H, 1, n, n)
        return M0.adjoint()


class MERouchon1(MERouchon):
    @lru_cache(maxsize=1)
    def M0(self, H_nh: Tensor, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I - 1j * dt * H_nh

    @lru_cache(maxsize=1)
    def M0inv(self, H_nh: Tensor, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I + 1j * dt * H_nh

    @lru_cache(maxsize=1)
    def M1s(self) -> Tensor:
        """First order Kraus operators."""
        # -> (1, len(jump_ops), n, n)
        return sqrt(self.dt) * self.jump_ops

    @lru_cache(maxsize=1)
    def M1sdag(self) -> Tensor:
        """Adjoint of the first order Kraus operators."""
        # -> (1, len(jump_ops), n, n)
        return self.M1s().adjoint()

    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
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
        M0 = self.M0(H_nh, dt)  # (b_H, 1, n, n)
        M1s = self.M1s()  # (1, len(jump_ops), n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, M1s)  # (b_H, b_rho, n, n)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
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
        M0 = self.M0(H_nh, dt)  # (b_H, 1, n, n)
        M0dag = self.M0dag(M0)  # (b_H, 1, n, n)
        M0inv = self.M0inv(H_nh, dt)  # (b_H, 1, n, n)
        M1s = self.M1s()  # (1, len(jump_ops), n, n)
        M1sdag = self.M1sdag()  # (1, len(jump_ops), n, n)

        # compute rho(t-dt)
        rho = kraus_map(rho, M0inv) - kraus_map(rho, M1s)
        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        phi = kraus_map(phi, M0dag) + kraus_map(phi, M1sdag)

        return rho, phi


class MERouchon1_5(MERouchon):
    # TODO implement caching
    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
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
        H_nh = self.H - 0.5j * self.sum_no_jump()  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_no_jump()  # (b_H, 1, n, n)
        # TODO Fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)  # (b_H, b_rho, n, n)

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
    @lru_cache(maxsize=1)
    def M0(self, H_nh: Tensor, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh

    @lru_cache(maxsize=1)
    def M0inv(self, H_nh: Tensor, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I + 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh

    @lru_cache(maxsize=1)
    def M1s(self, M0: Tensor, dt: float) -> Tensor:
        """First order Kraus operators."""
        # -> (b_H, len(jump_ops), n, n)
        return 0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)

    @lru_cache(maxsize=1)
    def M1sdag(self, M1s: Tensor) -> Tensor:
        """Adjoint of the first order Kraus operators."""
        # -> (b_H, len(jump_ops), n, n)
        return M1s.adjoint()

    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
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
        M0 = self.M0(H_nh, dt)  # (b_H, 1, n, n)
        M1s = self.M1s(M0, dt)  # (b_H, len(jump_ops), n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)  # (b_H, b_rho, n, n)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real  # (b_H, b_rho, n, n)

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
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
        M0 = self.M0(H_nh, dt)
        M0dag = self.M0dag(M0)
        M0inv = self.M0inv(H_nh, dt)
        M1s = self.M1s(M0, dt)
        M1sdag = self.M1sdag(M1s)

        # compute rho(t-dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0inv) - tmp + 0.5 * kraus_map(tmp, M1s)
        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        tmp = kraus_map(phi, M1sdag)
        phi = kraus_map(phi, M0dag) + tmp + 0.5 * kraus_map(tmp, M1sdag)

        return rho, phi
