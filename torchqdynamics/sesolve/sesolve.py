from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..odeint import odeint
from ..solver_options import Euler, SolverOption
from ..types import OperatorLike, TDOperatorLike, TensorLike, to_tensor
from .euler import SEEuler


def sesolve(
    H: TDOperatorLike,
    psi0: OperatorLike,
    t_save: TensorLike,
    *,
    save_states: bool = True,
    exp_ops: Optional[List[OperatorLike]] = None,
    solver: Optional[SolverOption] = None,
    gradient_alg: Optional[Literal['autograd', 'adjoint']] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
) -> Tuple[Tensor, Tensor]:
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

    # convert H to a tensor and batch by default
    H = to_tensor(H)
    H_batched = H[None, ...] if H.dim() == 2 else H

    # convert psi0 to a tensor and batch by default
    # TODO add test to check that psi0 has the correct shape
    b_H = H_batched.size(0)
    psi0 = to_tensor(psi0)
    psi0_batched = psi0[None, ...] if psi0.dim() == 2 else psi0
    psi0_batched = psi0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_psi0, n, 1)

    t_save = torch.as_tensor(t_save)

    exp_ops = to_tensor(exp_ops)

    if solver is None:
        solver = Euler(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Euler):
        qsolver = SEEuler(H_batched, solver)
    else:
        raise NotImplementedError

    # compute the result
    y_save, exp_save = odeint(
        qsolver, psi0_batched, t_save, exp_ops, save_states, gradient_alg, parameters
    )

    # restore correct batching
    if psi0.dim() == 2:
        y_save = y_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        y_save = y_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return y_save, exp_save
