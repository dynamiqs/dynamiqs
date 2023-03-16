from abc import abstractmethod
from typing import List

import torch

from .rouchon import Rouchon1Solver
from ..odeint import odeint, AutoDiffAlgorithm
from ..solver import Rouchon


class ForwardQSolver:
    @abstractmethod
    def forward(self, t, dt, rho):
        pass

    @abstractmethod
    def forward_adjoint(self, t, dt, phi):
        pass


def mesolve(
    H: torch.Tensor, rho0: torch.Tensor, t_save: torch.Tensor, solver,
    jump_ops: List[torch.Tensor] = None, exp_ops: List[torch.Tensor] = None,
    compute_gradient=True, save_states=True
):
    jump_ops = jump_ops or []
    exp_ops = exp_ops or []

    if solver is None:
        solver = Rouchon(dt=min(t_save[:-1] - t_save[1:]))

    autodiff_algorithm = AutoDiffAlgorithm.AUTOGRAD if compute_gradient else AutoDiffAlgorithm.NONE

    # define the QSolver
    qsolver = None
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = Rouchon1Solver(solver, H, jump_ops)

    if qsolver is None:
        raise NotImplementedError

    # compute the result
    return odeint(qsolver, rho0, t_save, exp_ops, save_states, autodiff_algorithm)
