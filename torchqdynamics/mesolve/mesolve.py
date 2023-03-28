from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..odeint import odeint
from ..solver_options import Euler, SolverOption
from ..types import OperatorLike, TDOperatorLike, TensorLike, to_tensor
from ..utils import is_ket, ket_to_dm
from .euler import MEEuler
from .rouchon import MERouchon1, MERouchon1_5, MERouchon2
from .solver_options import Rouchon1, Rouchon1_5, Rouchon2


def mesolve(
    H: TDOperatorLike,
    jump_ops: List[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    *,
    save_states: bool = True,
    exp_ops: Optional[List[OperatorLike]] = None,
    solver: Optional[SolverOption] = None,
    gradient_alg: Optional[Literal['autograd', 'adjoint']] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
) -> Tuple[Tensor, Tensor]:
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
      - `Rouchon1` (alias of `Rouchon`)
      - `Rouchon1_5`
      - `Rouchon2`

    Args:
        H (Tensor or Callable): Hamiltonian.
            Can be a tensor of shape (n, n) or (b_H, n, n) if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=t_save[-1]`.
        jump_ops (list of Tensor): List of jump operators.
            Each jump operator should be a tensor of shape (n, n).
        rho0 (Tensor): Initial density matrix.
            Tensor of shape (n, n) or (b_rho, n, n) if batched.
        t_save (Tensor, np.ndarray or list): Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`.
        save_states (bool, optional): If `True`, the density matrix is saved at every
            time value in `t_save`. If `False`, only the final density matrix is
            stored and returned. Defaults to `True`.
        exp_ops (list of Tensor, optional): List of operators for which the expectation
            value is computed at every time value in `t_save`.
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
    H_batched = H[None, ...] if H.dim() == 2 else H

    # convert jump_ops to a tensor
    if len(jump_ops) == 0:
        raise ValueError('Argument `jump_ops` must be a non-empty list of tensors.')
    jump_ops = to_tensor(jump_ops)

    # convert rho0 to a tensor and density matrix and batch by default
    rho0 = to_tensor(rho0)
    if is_ket(rho0):
        rho0 = ket_to_dm(rho0)
    b_H = H_batched.size(0)
    rho0_batched = rho0[None, ...] if rho0.dim() == 2 else rho0
    rho0_batched = rho0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_rho0, n, n)

    t_save = torch.as_tensor(t_save)

    exp_ops = to_tensor(exp_ops)

    if solver is None:
        # TODO Replace by adaptive time step solver when implemented.
        solver = Rouchon1(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Rouchon1):
        qsolver = MERouchon1(H_batched, jump_ops, solver)
    elif isinstance(solver, Rouchon1_5):
        qsolver = MERouchon1_5(H_batched, jump_ops, solver)
    elif isinstance(solver, Rouchon2):
        qsolver = MERouchon2(H_batched, jump_ops, solver)
    elif isinstance(solver, Euler):
        qsolver = MEEuler(H_batched, jump_ops, solver)
    else:
        raise NotImplementedError

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
