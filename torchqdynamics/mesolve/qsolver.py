from math import sqrt
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..odeint import AdjointQSolver
from ..solver_options import SolverOption
from ..solver_utils import inv_sqrtm, kraus_map, lindbladian
from ..types import TDOperator
from ..utils import trace


class MEAdaptive(ForwardQSolver):
    def __init__(self, H: TDOperator, jump_ops: Tensor, solver_options: SolverOption):
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)
        self.options = solver_options

    def forward(self, t: float, rho: Tensor) -> Tensor:
        """Compute drho / dt = L(rho) at time t."""
        return lindbladian(rho, self.H, self.jump_ops)


class MERouchon(AdjointQSolver):
    def __init__(self, H: TDOperator, jump_ops: Tensor, solver_options: SolverOption):
        """
        Args:
            H (): Hamiltonian, of shape (b_H, n, n).
            jump_ops (Tensor): Jump operators.
            solver_options ():
        """
        # convert H and jump_ops to sizes compatible with (b_H, len(jump_ops), n, n)
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)
        self.n = H.shape[-1]
        self.I = torch.eye(self.n).to(H)  # (n, n)
        self.options = solver_options


class MERouchon1(MERouchon):
    def forward(self, t: float, rho: Tensor) -> Tensor:
        """Compute rho(t+dt) using a Rouchon method of order 1.

        Args:
            t (float): Time.
            rho (Tensor): Density matrix of shape (b_H, b_rho, n, n).

        Returns:
            Density matrix at next time step, as tensor of shape (b_H, b_rho, n, n).
        """
        # get time step
        dt = self.options.dt

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        M1s = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor, parameters: Tuple[nn.Parameter, ...]
    ):
        """Compute rho(t-dt) and phi(t-dt) using a Rouchon method of order 1."""
        # get time step
        dt = self.options.dt

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump
        Hdag_nh = H_nh.adjoint()

        # compute rho(t-dt)
        M0 = self.I + 1j * dt * H_nh
        M1s = sqrt(dt) * self.jump_ops
        rho = kraus_map(rho, M0) - kraus_map(rho, M1s)
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        M0_adj = self.I + 1j * dt * Hdag_nh
        Ms_adj = torch.cat((M0_adj[None, ...], sqrt(dt) * self.jump_ops.adjoint()))
        phi = kraus_map(phi, Ms_adj)

        return rho, phi


class MERouchon1_5(MERouchon):
    def forward(self, t: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 1.5.

        Note:
            No need for trace renormalization since the scheme is trace-preserving
            by construction.

        Args:
            t (float): Time.
            rho (Tensor): Density matrix of shape (b_H, b_rho, n, n).

        Returns:
            Density matrix at next time step, as tensor of shape (b_H, b_rho, n, n).
        """
        # get time step
        dt = self.options.dt

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_nojump  # (b_H, 1, n, n)
        # TODO Fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor, parameters: Tuple[nn.Parameter, ...]
    ):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t: float, rho: Tensor):
        r"""Compute rho(t+dt) using a Rouchon method of order 2.

        Note:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.

        Args:
            t (float): Time.
            rho (Tensor): Density matrix of shape (b_H, b_rho, n, n).

        Returns:
            Density matrix at next time step, as tensor of shape (b_H, b_rho, n, n).
        """
        # get time step
        dt = self.options.dt

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh  # (b_H, 1, n, n)
        M1s = 0.5 * sqrt(dt) * (
            self.jump_ops @ M0 + M0 @ self.jump_ops
        )  # (b_H, len(jump_ops), n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor, parameters: Tuple[nn.Parameter, ...]
    ):
        """Compute rho(t-dt) and phi(t-dt) using a Rouchon method of order 2."""
        # get time step
        dt = self.options.dt

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump
        Hdag_nh = H_nh.adjoint()

        # compute rho(t-dt)
        M0 = self.I + 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh
        M1s = 0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0) - tmp + 0.5 * kraus_map(tmp, M1s)
        rho = rho / trace(rho)[..., None, None].real

        # compute phi(t-dt)
        M0_adj = self.I + 1j * dt * Hdag_nh - 0.5 * dt**2 * Hdag_nh @ Hdag_nh
        M1s_adj = 0.5 * sqrt(dt) * (
            self.jump_ops.adjoint() @ M0_adj + M0_adj @ self.jump_ops.adjoint()
        )
        tmp = kraus_map(phi, M1s_adj)
        phi = kraus_map(phi, M0_adj) + tmp + 0.5 * kraus_map(tmp, M1s_adj)

        return rho, phi
