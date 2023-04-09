from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..odeint import odeint
from ..solver_options import AdaptiveStep, Dopri45, Euler, SolverOption
from ..types import OperatorLike, TDOperatorLike, TensorLike, to_tensor
from .adaptive import SEAdaptive
from .euler import SEEuler


def sesolve(
    H: TDOperatorLike,
    psi0: OperatorLike,
    t_save: TensorLike,
    *,
    save_states: bool = True,
    exp_ops: OperatorLike | list[OperatorLike] | None = None,
    solver: SolverOption | None = None,
    gradient_alg: Literal['autograd', 'adjoint'] | None = None,
    parameters: tuple[nn.Parameter, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    # Args:
    #     H: (b_H?, n, n)
    #     psi0: (b_psi0?, n, 1)
    #
    # Returns:
    #     (y_save, exp_save) with
    #     - y_save: (b_H?, b_psi0?, len(t_save), n, 1)
    #     - exp_save: (b_H?, b_psi0?, len(exp_ops), len(t_save))

    # TODO support density matrices too
    # TODO H is assumed to be time-independent from here (temporary)

    # default solver
    if solver is None:
        solver = Dopri45()

    # convert H to a tensor and batch by default
    H = to_tensor(H)
    H_batched = H[None, ...] if H.ndim == 2 else H

    # convert psi0 to a tensor and batch by default
    # TODO add test to check that psi0 has the correct shape
    b_H = H_batched.size(0)
    psi0 = to_tensor(psi0)
    psi0_batched = psi0[None, ...] if psi0.ndim == 2 else psi0
    psi0_batched = psi0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_psi0, n, 1)

    # convert t_save to tensor
    t_save = torch.as_tensor(t_save)

    # convert exp_ops to tensor
    exp_ops = to_tensor(exp_ops)
    exp_ops = exp_ops[None, ...] if exp_ops.ndim == 2 else exp_ops

    # define the QSolver
    args = (H_batched, solver)
    if isinstance(solver, Euler):
        qsolver = SEEuler(*args)
    elif isinstance(solver, AdaptiveStep):
        qsolver = SEAdaptive(*args)
    else:
        raise NotImplementedError(f'Solver {type(solver)} is not implemented.')

    # compute the result
    y_save, exp_save = odeint(
        qsolver,
        psi0_batched,
        t_save,
        save_states=save_states,
        exp_ops=exp_ops,
        gradient_alg=gradient_alg,
        parameters=parameters,
    )

    # restore correct batching
    if psi0.dim() == 2:
        y_save = y_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        y_save = y_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return y_save, exp_save
