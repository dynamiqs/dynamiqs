from __future__ import annotations

from abc import abstractmethod
from functools import lru_cache
from math import sqrt

import torch
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
    def M0(self, t: float, dt: float) -> Tensor:
        r"""Compute the zero-th order Kraus operator at time $t$.

        Args:
            t: Time.
            dt: Time step.

        Returns:
            Zero-th order Kraus operator of shape `(b_H, 1, n, n)`.
        """
        pass

    @lru_cache(maxsize=1)
    def M0_adj(self, t: float, dt: float) -> Tensor:
        r"""Compute the adjoint of the zero-th order Kraus operator at time $t$.

        Args:
            t: Time.
            dt: Time step.

        Returns:
            Adjoint of the zero-th order Kraus operator of shape `(b_H, 1, n, n)`.
        """
        return self.M0(t, dt).adjoint()

    @abstractmethod
    def M1s(self, t: float, dt: float) -> Tensor:
        r"""Compute the first order Kraus operator at time $t$.

        Args:
            t: Time.
            dt: Time step.

        Returns:
            First order Kraus operator of shape `(b_H, 1, n, n)`.
        """
        pass

    @lru_cache(maxsize=1)
    def M1s_adj(self, t: float, dt: float) -> Tensor:
        r"""Compute the adjoint of the first order Kraus operator at time $t$.

        Args:
            t: Time.
            dt: Time step.

        Returns:
            First order Kraus operator of shape `(b_H, 1, n, n)`.
        """
        return self.M1s(t, dt).adjoint()


class MERouchon1(MERouchon):
    @lru_cache(maxsize=1)
    def M0(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I - 1j * dt * self.H_nh(t)

    @lru_cache(maxsize=1)
    def M1s(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return sqrt(dt) * self.jump_ops

    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, self.M0(t, dt)) + kraus_map(rho, self.M1s(t, dt))
        # rho: (b_H, b_rho, n, n)

        # normalize by the trace
        return rho / trace(rho)[..., None, None].real

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 1.

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

        # compute rho(t-dt)
        rho = kraus_map(rho, self.M0(t, -dt)) - kraus_map(rho, self.M1s(t, dt))
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        phi = kraus_map(phi, self.M0_adj(t, dt)) + kraus_map(phi, self.M1s_adj(t, dt))

        return rho, phi


class MERouchon1_5(MERouchon):
    @lru_cache(maxsize=1)
    def M0(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return self.I - 1j * dt * self.H_nh(t)

    @lru_cache(maxsize=1)
    def M1s(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return sqrt(dt) * self.jump_ops

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

        # build normalization matrix
        S = (
            self.M0_adj(t, dt) @ self.M0(t, dt) + dt * self.sum_no_jump
        )  # (b_H, 1, n, n)

        # TODO Fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)  # (b_H, b_rho, n, n)
        rho = kraus_map(rho, self.M0(t, dt)) + kraus_map(
            rho, self.M1s(t, dt)
        )  # (b_H, b_rho, n, n)

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
    @lru_cache(maxsize=1)
    def M0(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        H_nh = self.H_nh(t)
        return self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh

    @lru_cache(maxsize=1)
    def M1s(self, t: float, dt: float) -> Tensor:
        # -> (b_H, 1, n, n)
        M0 = self.M0(t, dt)
        return 0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)

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

        # compute rho(t+dt)
        tmp = kraus_map(rho, self.M1s(t, dt))  # (b_H, b_rho, n, n)
        rho = (
            kraus_map(rho, self.M0(t, dt)) + tmp + 0.5 * kraus_map(tmp, self.M1s(t, dt))
        )  # (b_H, b_rho, n, n)

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

        # compute rho(t-dt)
        tmp = kraus_map(rho, self.M1s(t, dt))
        rho = (
            kraus_map(rho, self.M0(t, -dt))
            - tmp
            + 0.5 * kraus_map(tmp, self.M1s(t, dt))
        )
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        tmp = kraus_map(phi, self.M1s_adj(t, dt))
        phi = (
            kraus_map(phi, self.M0_adj(t, dt))
            + tmp
            + 0.5 * kraus_map(tmp, self.M1s_adj(t, dt))
        )

        return rho, phi
