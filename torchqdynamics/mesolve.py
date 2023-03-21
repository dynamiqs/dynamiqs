from math import sqrt
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .odeint import AdjointQSolver, odeint
from .solver import Rouchon, SolverOption
from .solver_utils import inv_sqrtm, kraus_map
from .utils import trace


def mesolve(
    H: Union[torch.Tensor, Callable[[float], torch.Tensor]],
    jump_ops: List[torch.Tensor],
    rho0: torch.Tensor,
    t_save: torch.Tensor,
    *,
    exp_ops: Optional[List[torch.Tensor]] = None,
    save_states: bool = True,
    gradient_alg: Optional[str] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
    solver: Optional[SolverOption] = None,
):
    if isinstance(t_save, (list, np.ndarray)):
        t_save = torch.tensor(t_save)
    if exp_ops is None:
        exp_ops = []
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
    return odeint(qsolver, rho0, t_save, exp_ops, save_states, gradient_alg)


class MERouchon(AdjointQSolver):
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
