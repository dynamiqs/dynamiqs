import torch

from ..solver import Rouchon
from . import ForwardQSolver


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


class Rouchon1Solver(RouchonSolver):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 1."""
        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.jump_ops_sum

        # build time-dependent Kraus operator
        M0 = self.I - 1j * dt * H_nh

        # compute rho(t+dt)
        rho = (
            M0 @ rho @ M0.adjoint() + dt *
            (self.jump_ops @ rho.unsqueeze(0) @ self.jump_ops_sum).sum(dim=0)
        )
        rho = rho / rho.trace()
        return rho

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError
