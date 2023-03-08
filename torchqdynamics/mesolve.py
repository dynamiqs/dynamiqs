import torch

from .odeint import odeint
from .solver import Rouchon


def mesolve(
    H, jump_ops, rho0, tsave, solver=None, sensitivity='autograd', variables=None
):
    if solver is None:
        # TODO: The default dt should not be choosen in such an arbitrary
        # fashion, which depends on the time unit used by the user.
        solver = Rouchon(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = MERouchon1(H, jump_ops, solver)
        elif solver.order == 2:
            qsolver = MERouchon2(H, jump_ops, solver)
    else:
        raise NotImplementedError

    # compute the result
    return odeint(qsolver, rho0, tsave, sensitivity=sensitivity, variables=variables)


class QSolver:
    def __init__(self):
        pass

    def forward(self, t, dt, rho):
        pass

    def forward_adjoint(self, t, dt, phi):
        pass


class MERouchon(QSolver):
    def __init__(self, H, jump_ops, solver_options):
        self.H = H
        self.jump_ops = jump_ops
        self.jumpdag_ops = jump_ops.adjoint()
        self.sum_nojump = (self.jumpdag_ops @ self.jump_ops).sum(dim=0)
        self.I = torch.eye(H(0).shape[-1]).to(H(0))
        self.options = solver_options


class MERouchon1(MERouchon):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 1."""
        # mon-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # build time-dependent Kraus operator
        M0 = self.I - 1j * dt * H_nh

        # compute rho(t+dt)
        rho = M0 @ rho @ M0.adjoint()
        rho += dt * (self.jump_ops @ rho.unsqueeze(0) @ self.jumpdag_ops).sum(dim=0)
        return rho

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t, dt, rho):
        raise NotImplementedError

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError
