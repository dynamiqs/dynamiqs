from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from ..odeint import odeint
from ..solver_options import SolverOption
from ..types import OperatorLike, TensorLike, TimeDependentOperatorLike, to_tensor


def sesolve(
    H: TimeDependentOperatorLike,
    psi: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: Optional[List[OperatorLike]] = None,
    save_states: bool = True,
    gradient_alg: Literal[None, 'autograd', 'adjoint'] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
    solver: Optional[SolverOption] = None,
) -> torch.Tensor:
    # Args:
    #     H: (b_H?, n, n)
    #     psi: (b_psi?, n, 1)
    #
    # Returns:
    #     (y_save, exp_save) with
    #     - y_save: (b_H?, b_psi?, len(t_save), n, 1)
    #     - exp_save: (b_H?, b_psi?, len(exp_ops), len(t_save))

    # TODO: support density matrices too
    # TODO: H is assumed to be time-independent from here (temporary)

    # convert H to a tensor and batch by default
    H = to_tensor(H)
    H_batched = H[None, ...] if H.dim() == 2 else H

    # convert psi to a tensor and batch by default
    psi = to_tensor(psi)
    psi_batched = psi[None, ...] if psi.dim() == 2 else psi

    t_save = torch.as_tensor(t_save)

    exp_ops = to_tensor(exp_ops)

    # TODO: placeholder, to remove
    if solver is None:
        pass

    # TODO: placeholder, to remove
    qsolver = None

    # compute the result
    b_H = H_batched.size(0)
    y0 = psi_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_psi, n, n)
    y_save, exp_save = odeint(qsolver, y0, t_save, exp_ops, save_states, gradient_alg)

    # restore correct batching
    if psi.dim() == 2:
        y_save = y_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        y_save = y_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return y_save, exp_save
