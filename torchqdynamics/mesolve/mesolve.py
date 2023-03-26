from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from ..odeint import odeint
from ..solver_options import SolverOption
from ..types import OperatorLike, TensorLike, TimeDependentOperatorLike, to_tensor
from ..utils import is_ket, ket_to_dm
from .rouchon import MERouchon1, MERouchon1_5, MERouchon2
from .solver_options import Rouchon1, Rouchon1_5, Rouchon2


def mesolve(
    H: TimeDependentOperatorLike,
    jump_ops: List[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: Optional[List[OperatorLike]] = None,
    save_states: bool = True,
    gradient_alg: Literal[None, 'autograd', 'adjoint'] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
    solver: Optional[SolverOption] = None,
):
    # Args:
    #     H: (b_H?, n, n)
    #     rho0: (b_rho0?, n, n)
    #
    # Returns:
    #     (y_save, exp_save) with
    #     - y_save: (b_H?, b_rho0?, len(t_save), n, n)
    #     - exp_save: (b_H?, b_rho0?, len(exp_ops), len(t_save))

    # TODO: H is assumed to be time-independent from here (temporary)

    # convert H to a tensor and batch by default
    H = to_tensor(H)
    H_batched = H[None, ...] if H.dim() == 2 else H

    # convert jump_ops to a tensor
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list of torch.Tensor.'
        )
    jump_ops = to_tensor(jump_ops)

    # convert rho0 to a tensor and density matrix and batch by default
    rho0 = to_tensor(rho0)
    if is_ket(rho0):
        rho0 = ket_to_dm(rho0)
    rho0_batched = rho0[None, ...] if rho0.dim() == 2 else rho0

    t_save = torch.as_tensor(t_save)

    exp_ops = to_tensor(exp_ops)

    if solver is None:
        # TODO: Replace by adaptive time step solver when implemented.
        solver = Rouchon1(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Rouchon1):
        qsolver = MERouchon1(H_batched, jump_ops, solver)
    elif isinstance(solver, Rouchon1_5):
        qsolver = MERouchon1_5(H_batched, jump_ops, solver)
    elif isinstance(solver, Rouchon2):
        qsolver = MERouchon2(H_batched, jump_ops, solver)
    else:
        raise NotImplementedError

    # compute the result
    b_H = H_batched.size(0)
    y0 = rho0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_rho0, n, n)
    y_save, exp_save = odeint(qsolver, y0, t_save, exp_ops, save_states, gradient_alg)

    # restore correct batching
    if rho0.dim() == 2:
        y_save = y_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        y_save = y_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return y_save, exp_save
