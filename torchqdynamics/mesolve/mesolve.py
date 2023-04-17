from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..odeint import odeint
from ..solver_options import AdaptiveStep, Dopri45, Euler, SolverOption
from ..tensor_types import OperatorLike, TDOperatorLike, TensorLike, to_tensor
from ..utils import is_ket, ket_to_dm
from .adaptive import MEAdaptive
from .euler import MEEuler
from .rouchon import MERouchon1, MERouchon1_5, MERouchon2
from .solver_options import Rouchon1, Rouchon1_5, Rouchon2


def mesolve(
    H: TDOperatorLike,
    jump_ops: OperatorLike | list[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    *,
    save_states: bool = True,
    exp_ops: OperatorLike | list[OperatorLike] | None = None,
    solver: SolverOption | None = None,
    gradient_alg: Literal['autograd', 'adjoint'] | None = None,
    parameters: tuple[nn.Parameter, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Solve the Lindblad master equation for a Hamiltonian and set of jump operators.

    The Hamiltonian `H` and the initial density matrix `rho0` can be batched over to
    solve multiple master equations in a single run. The jump operators `jump_ops` and
    time list `t_save` are common to all batches.

    `mesolve` can be differentiated through using either the default PyTorch autograd
    library (`gradient_alg="autograd"`), or a custom adjoint state differentiation
    (`gradient_alg="adjoint"`). For the latter, a solver that is stable in the backward
    pass should be used (e.g. Rouchon solver). By default (if no gradient is required),
    the graph of operations is not stored for improved performance of the solver.

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Piecewise constant Hamiltonians can also be
    passed as... TODO Complete with full Hamiltonian format

    Available solvers:
      - `Dopri45`: Dormand-Prince of order 5. Default solver.
      - `Rouchon1`: Rouchon method of order 1. Alias of `Rouchon`.
      - `Rouchon1_5`: Rouchon method of order 1 with Kraus map trace renormalization.
      - `Rouchon2`: Rouchon method of order 2.
      - `Euler`: Euler method.

    Args:
        H (Tensor or Callable): Hamiltonian.
            Can be a tensor of shape `(n, n)` or `(b_H, n, n)` if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=t_save[-1]`.
        jump_ops (Tensor, or list of Tensors): List of jump operators.
            Each jump operator should be a tensor of shape `(n, n)`.
        rho0 (Tensor): Initial density matrix.
            Tensor of shape `(n, n)` or `(b_rho, n, n)` if batched.
        t_save (Tensor, np.ndarray or list): Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`.
        save_states (bool, optional): If `True`, the density matrix is saved at every
            time value in `t_save`. If `False`, only the final density matrix is
            stored and returned. Defaults to `True`.
        exp_ops (Tensor, or list of Tensors, optional): List of operators for which the
            expectation value is computed at every time value in `t_save`.
        solver (SolverOption, optional): Solver used to compute the master equation
            solutions. See the list of available solvers.
        gradient_alg (str, optional): Algorithm used for computing gradients in the
            backward pass. Defaults to `None`.
        parameters (tuple of nn.Parameter): Parameters with respect to which gradients
            are computed during the adjoint state backward pass.

    Returns:
        A tuple `(rho_save, exp_save)` where
            `rho_save` is a tensor with the computed density matrices at `t_save`
                times, and of shape `(len(t_save), n, n)` or `(b_H, b_rho, len(t_save),
                n, n)` if batched. If `save_states` is `False`, only the final density
                matrix is returned with the same shape as the initial input.
            `exp_save` is a tensor with the computed expectation values at `t_save`
                times, and of shape `(len(exp_ops), len(t_save))` or `(b_H, b_rho,
                len(exp_ops), len(t_save))` if batched.
    """
    # TODO H is assumed to be time-independent from here (temporary)

    # convert H to a tensor and batch by default
    H = to_tensor(H)
    H_batched = H[None, ...] if H.ndim == 2 else H

    # convert jump_ops to a tensor
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list of tensors. Otherwise,'
            ' consider using `sesolve`.'
        )
    jump_ops = to_tensor(jump_ops)
    jump_ops = jump_ops[None, ...] if jump_ops.ndim == 2 else jump_ops

    # convert rho0 to a tensor and density matrix and batch by default
    rho0 = to_tensor(rho0)
    if is_ket(rho0):
        rho0 = ket_to_dm(rho0)
    b_H = H_batched.size(0)
    rho0_batched = rho0[None, ...] if rho0.ndim == 2 else rho0
    rho0_batched = rho0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_rho0, n, n)

    # convert t_save to a tensor
    t_save = torch.as_tensor(t_save)

    # convert exp_ops to a tensor
    exp_ops = to_tensor(exp_ops)
    exp_ops = exp_ops[None, ...] if exp_ops.ndim == 2 else exp_ops

    # default solver
    if solver is None:
        solver = Dopri45()

    # define the QSolver
    args = (H_batched, jump_ops, solver)
    if isinstance(solver, Rouchon1):
        qsolver = MERouchon1(*args)
    elif isinstance(solver, Rouchon1_5):
        qsolver = MERouchon1_5(*args)
    elif isinstance(solver, Rouchon2):
        qsolver = MERouchon2(*args)
    elif isinstance(solver, AdaptiveStep):
        qsolver = MEAdaptive(*args)
    elif isinstance(solver, Euler):
        qsolver = MEEuler(*args)
    else:
        raise NotImplementedError(f'Solver {type(solver)} is not implemented.')

    # compute the result
    rho_save, exp_save = odeint(
        qsolver,
        rho0_batched,
        t_save,
        save_states=save_states,
        exp_ops=exp_ops,
        gradient_alg=gradient_alg,
        parameters=parameters,
    )

    # restore correct batching
    if rho0.dim() == 2:
        rho_save = rho_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        rho_save = rho_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return rho_save, exp_save
