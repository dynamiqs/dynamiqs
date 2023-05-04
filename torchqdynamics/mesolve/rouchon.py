from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..solvers.ode.forward_solver import AdjointForwardSolver
from ..solvers.solver import depends_on_H
from ..utils.solver_utils import inv_sqrtm, kraus_map
from ..utils.utils import trace


class MERouchon(AdjointForwardSolver):
    def __init__(self, *args, jump_ops: Tensor):
        """
        Args:
            H: Hamiltonian, of shape `(b_H, n, n)`.
            jump_ops: Jump operators, of shape `(len(jump_ops), n, n)`.
        """
        super().__init__(*args)

        self.n = self.H(0.0).shape[-1]
        self.I = torch.eye(self.n).to(self.H(0.0))  # (n, n)
        self.dt = self.options.dt

        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)

        self.M1s = sqrt(self.dt) * self.jump_ops  # (1, len(jump_ops), n, n)
        self.M1s_adj = sqrt(self.dt) * self.jump_ops.adjoint()


class MERouchon1(MERouchon):
    @depends_on_H
    def H_nh(self, t: float) -> Tensor:
        # non-hermitian Hamiltonian at time t
        return self.H(t) - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

    @depends_on_H
    def Hdag_nh(self, t: float) -> Tensor:
        return self.H_nh(t).adjoint()

    @depends_on_H
    def M0(self, t: float, dt: float) -> Tensor:
        # build time-dependent Kraus operators
        return self.I - 1j * dt * self.H_nh(t)  # (b_H, 1, n, n)

    @depends_on_H
    def M0_adj(self, t: float, dt: float) -> Tensor:
        return self.I + 1j * dt * self.Hdag_nh(t)

    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
        r"""Compute $\rho(t+dt)$ using a Rouchon method of order 1.

        Args:
            t: Time.
            rho: Density matrix of shape `(b_H, b_rho, n, n)`.

        Returns:
            Density matrix at next time step, as tensor of shape `(b_H, b_rho, n, n)`.
        """
        # compute rho(t+dt)
        rho = kraus_map(rho, self.M0(t, dt)) + kraus_map(rho, self.M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 1."""
        # non-hermitian Hamiltonian at time t

        # compute rho(t-dt)
        rho = kraus_map(rho, self.M0(t, dt)) - kraus_map(rho, self.M1s)
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        H_nh = self.H(t) - 0.5j * self.sum_nojump
        Hdag_nh = H_nh.adjoint()
        M0_adj = self.I + 1j * self.dt * Hdag_nh

        # TODO: using the cached version of M0_adj breaks the tests
        # if H_nh is marked as `@depends_on_H`
        # M0_adj = self.M0_adj(t)

        phi = kraus_map(phi, M0_adj) + kraus_map(phi, self.M1s_adj)

        return rho, phi


class MERouchon1_5(MERouchon):
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
        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * self.dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_nojump  # (b_H, 1, n, n)
        # TODO Fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
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
        # non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        # M0: (b_H, 1, n, n)
        M0 = self.I - 1j * dt * H_nh - 0.5 * self.dt**2 * H_nh @ H_nh
        M1s = (
            0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)
        )  # (b_H, len(jump_ops), n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(self, t: float, dt: float, rho: Tensor, phi: Tensor):
        r"""Compute $\rho(t-dt)$ and $\phi(t-dt)$ using a Rouchon method of order 2."""
        # non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump
        Hdag_nh = H_nh.adjoint()

        # compute rho(t-dt)
        M0 = self.I + 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh
        M1s = 0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0) - tmp + 0.5 * kraus_map(tmp, M1s)
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        M0_adj = self.I + 1j * dt * Hdag_nh - 0.5 * dt**2 * Hdag_nh @ Hdag_nh
        M1s_adj = (
            0.5
            * sqrt(dt)
            * (self.jump_ops.adjoint() @ M0_adj + M0_adj @ self.jump_ops.adjoint())
        )
        tmp = kraus_map(phi, M1s_adj)
        phi = kraus_map(phi, M0_adj) + tmp + 0.5 * kraus_map(tmp, M1s_adj)

        return rho, phi
