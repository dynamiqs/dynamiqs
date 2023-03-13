from typing import List

import numpy as np
import torch

from torchqdynamics.mesolve.rouchon import Rouchon1Solver, Rouchon2Solver

from .odeint import GradientAlgorithm, odeint
from .solver import Rouchon


def mesolve(
    H: torch.Tensor, rho0: torch.Tensor, t_save: torch.Tensor, solver,
    t_step: torch.Tensor = None, jump_ops: List[torch.Tensor] = None,
    exp_ops: List[torch.Tensor] = None, compute_gradient=True, save_states=True
):
    jump_ops = jump_ops or []
    exp_ops = exp_ops or []
    t_step = t_step or t_save

    if solver is None:
        solver = Rouchon(dt=np.min(t_step[:-1] - t_step[1:]))

    gradient_algorithm = GradientAlgorithm.AUTOGRAD if compute_gradient else GradientAlgorithm.NONE

    # define the QSolver
    qsolver = None
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = Rouchon1Solver(solver, H, jump_ops)
        elif solver.order == 2:
            qsolver = Rouchon2Solver(solver, H, jump_ops)
    else:
        raise NotImplementedError

    # compute the result
    return odeint(
        qsolver, rho0, t_save, t_step, exp_ops, save_states, gradient_algorithm
    )
