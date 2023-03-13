import numpy as np
import torch

from torchqdynamics.odeint import ForwardQSolver
from torchqdynamics.solver import Rouchon


class RouchonSolver(ForwardQSolver):
    def __init__(self, options: Rouchon, H, jump_ops):
        self.H = H

        self.jump_ops = jump_ops
        self.jump_ops_dag = jump_ops.adjoint()
        self.jump_ops_sum = (self.jump_ops_dag @ self.jump_ops).sum(dim=0)

        self.I = torch.eye(H(0).shape[-1]).to(H(0))
        self.options = options

        self._M0_dt = None
        self._M0 = None
        self._H_prev_time_index = None

    def H(self, t):
        if isinstance(self.H, torch.Tensor):
            return self.H, False
        elif isinstance(self.H, dict):
            times = np.array(list(self.H.keys()))
            time_index = np.where(times[:-1] < t < times[1:])

            H_has_changed = time_index == self._H_prev_time_index

            return self.H[time_index], H_has_changed
        elif callable(self.H):
            self._H_has_changed = True
            return self.H(t), True
        else:
            raise ValueError(
                'H must be a tensor, a dictionnary of (time, tensor) or a callable'
            )

    def M0(self, H, dt, H_has_changed):
        if not H_has_changed and dt == self._M0_dt:
            return self._M0

        self._M0 = self.compute_M0(H, dt)
        return self._M0

    def compute_M0(self, _H, _dt):
        return None


class Rouchon1Solver(RouchonSolver):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 1."""
        # non-hermitian Hamiltonian at time t
        H, H_has_changed = self.H(t)
        H_nh = H - 0.5j * self.jump_ops_sum

        # build time-dependent Kraus operator
        M0 = self.M0(H_nh, dt, H_has_changed)

        # compute rho(t+dt)
        rho = (
            M0 @ rho @ M0.adjoint() + dt *
            (self.jump_ops @ rho.unsqueeze(0) @ self.jump_ops_dag).sum(dim=0)
        )
        rho = rho / rho.trace()
        return rho

    def compute_M0(self, H, dt):
        return self.I - 1j * dt * H


class Rouchon2Solver(RouchonSolver):
    def __init__(self, options: Rouchon, H, jump_ops):
        super().__init__(options, H, jump_ops)

        # non-hermitian Hamiltonian at time t

    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 2.
        NB: For fast time-varying Hamiltonians, this method is not order 2 because the
        second-order time derivative term is neglected. This term should be added in the
        zero-th order Kraus operator, as M0 += -0.5j * dt**2 * \dot{H}.
        """

        H, H_has_changed = self.H(t)
        H_nh = H - 0.5j * self.jump_ops_sum
        M0 = self.M0(H_nh, dt, H_has_changed)
        M1s = 0.5 * (self.jump_ops @ M0 + M0 @ self.jump_ops)
        # build time-dependent Kraus operators

        # compute rho(t+dt)
        tmp = dt * (M1s @ rho.unsqueeze(0) @ M1s.adjoint()).sum(dim=0)
        rho = (
            M0 @ rho @ M0.adjoint() + tmp + 0.5 * dt *
            (M1s @ rho.unsqueeze(0) @ M1s.adjoint()).sum(dim=0)
        )
        rho = rho / rho.trace()
        return rho

    def compute_M0(self, H_nh, dt):
        return self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh


def inv_sqrtm(A: torch.Tensor) -> torch.Tensor:
    """Compute the inverse square root of a matrix using its eigendecomposition.
    TODO: Replace with Schur decomposition once released by PyTorch.
    See the feature request at https://github.com/pytorch/pytorch/issues/78809.
    """
    vals, vecs = torch.linalg.eigh(A)
    return vecs @ torch.linalg.solve(vecs, torch.diag(vals**(-0.5)), left=False)
