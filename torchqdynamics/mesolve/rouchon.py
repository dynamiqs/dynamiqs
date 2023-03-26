from math import sqrt

import torch
from torch import Tensor

from ..odeint import AdjointQSolver
from ..solver_options import SolverOption
from ..solver_utils import inv_sqrtm, kraus_map
from ..types import TimeDependentOperator
from ..utils import trace


class MERouchon(AdjointQSolver):
    def __init__(
        self, H: TimeDependentOperator, jump_ops: Tensor, solver_options: SolverOption
    ):
        # Args:
        #     H: (b_H, n, n)

        # convert H and jump_ops to sizes compatible with (b_H, len(jump_ops), n, n)
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)
        n = H.shape[-1]
        self.I = torch.eye(n).to(H)  # (n, n)
        self.options = solver_options


class MERouchon1(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 1."""
        # Args:
        #     rho: (b_H, b_rho0, n, n)
        #
        # Returns:
        #     (b_H, b_rho0, n, n)

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

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError


class MERouchon1_5(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 1.5."""
        # Args:
        #     rho: (b_H, b_rho0, n, n)
        #
        # Returns:
        #     (b_H, b_rho0, n, n)

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_nojump  # (b_H, 1, n, n)
        # TODO: fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)

        # no need to normalize by the trace because this scheme is trace
        # preserving by construction

        return rho

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 2.

        NOTE: For fast time-varying Hamiltonians, this method is not order 2 because the
        second-order time derivative term is neglected. This term could be added in the
        zero-th order Kraus operator if needed, as M0 += -0.5j * dt**2 * \dot{H}.
        """
        # Args:
        #     rho: (b_H, b_rho0, n, n)
        #
        # Returns:
        #     (b_H, b_rho0, n, n)

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

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError
