from math import sqrt

import torch

from .odeint import odeint
from .solver import Rouchon


def mesolve(
    H, jump_ops, rho0, tsave, solver=None, sensitivity='autograd', parameters=None
):
    if solver is None:
        # TODO: Replace by adaptive time step solver when implemented.
        solver = Rouchon(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = MERouchon1(H, jump_ops, solver)
        elif solver.order == 1.5:
            qsolver = MERouchon1_5(H, jump_ops, solver)
        elif solver.order == 2:
            qsolver = MERouchon2(H, jump_ops, solver)
    else:
        raise NotImplementedError

    # compute the result
    return odeint(qsolver, rho0, tsave)


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
        # non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh
        Ms = torch.cat((M0[None, ...], sqrt(dt) * self.jump_ops))

        # compute rho(t+dt)
        rho = kraus_map(rho, Ms)
        return rho / trace(rho)[..., None, None].real

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


class MERouchon1_5(MERouchon):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 1.5."""
        # non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh
        Ms = torch.cat((M0[None, ...], sqrt(dt) * self.jump_ops))

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_nojump
        S_inv_sqrtm = inv_sqrtm(S)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm[None, ...])
        rho = kraus_map(rho, Ms)
        return rho

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 2.

        NOTE: For fast time-varying Hamiltonians, this method is not order 2 because the
        second-order time derivative term is neglected. This term could be added in the
        zero-th order Kraus operator if needed, as M0 += -0.5j * dt**2 * \dot{H}.
        """
        # non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh
        M1s = 0.5 * sqrt(dt) * (self.jump_ops @ M0 + M0 @ self.jump_ops)

        # compute rho(t+dt)
        rho_ = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0[None, ...]) + rho_ + 0.5 * kraus_map(rho_, M1s)
        return rho / trace(rho)[..., None, None].real

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


def kraus_map(rho, operators):
    """Compute the application of a Kraus map on an input density matrix.

    This is equivalent to `torch.sum(operators @ rho[None,...] @ operators.adjoint(),
    dim=0)`. The use of einsum yields better performances on large matrices, but may
    cause a small overhead on smaller matrices (N <~ 50).

    Args:
        rho: Density matrix of shape (..., n, n).
        operators: Kraus operators of shape (b, n, n).
    Returns:
        Density matrix of shape (..., n, n) with the Kraus map applied.
    """
    return torch.einsum('mij,...jk,mkl->...il', operators, rho, operators.adjoint())


def trace(rho):
    """Compute the batched trace of a tensor over its last two dimensions."""
    return torch.einsum('...ii', rho)


def inv_sqrtm(mat):
    """Compute the inverse square root of a matrix using its eigendecomposition.

    TODO: Replace with Schur decomposition once released by PyTorch.
    See the feature request at https://github.com/pytorch/pytorch/issues/78809.
    Alternatively, see
    https://github.com/pytorch/pytorch/issues/25481#issuecomment-584896176
    for sqrtm implementation.
    """
    vals, vecs = torch.linalg.eigh(mat)
    return vecs @ torch.linalg.solve(vecs, torch.diag(vals**(-0.5)), left=False)
