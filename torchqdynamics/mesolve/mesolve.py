from __future__ import annotations

import torch
from torch import Tensor

from ..options import (
    Dopri5,
    Euler,
    ODEAdaptiveStep,
    Options,
    Rouchon1,
    Rouchon1_5,
    Rouchon2,
)
from ..tensor_formatter import TensorFormatter
from ..utils.tensor_types import (
    OperatorLike,
    TDOperatorLike,
    TensorLike,
    dtype_complex_to_real,
)
from .adaptive import MEAdaptive
from .euler import MEEuler
from .rouchon import MERouchon1, MERouchon1_5, MERouchon2


def mesolve(
    H: TDOperatorLike,
    jump_ops: OperatorLike | list[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: OperatorLike | list[OperatorLike] | None = None,
    options: Options | None = None,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Solve the Lindblad master equation for a Hamiltonian and set of jump operators.

    The Hamiltonian `H` and the initial density matrix `rho0` can be batched over to
    solve multiple master equations in a single run. The jump operators `jump_ops` and
    time list `t_save` are common to all batches.

    `mesolve` can be differentiated through using either the default PyTorch autograd
    library (`gradient_alg="autograd"`), or a custom adjoint state differentiation
    (`gradient_alg="adjoint"`). For the latter, a solver that is stable in the backward
    pass should be used (e.g. Rouchon solver). By default (if no gradient is required),
    the graph of operations is not stored to improve performance.

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Piecewise constant Hamiltonians can also be
    passed as... TODO Complete with full Hamiltonian format

    Available solvers:
      - `Dopri5`: Dormand-Prince of order 5. Default solver.
      - `Rouchon1`: Rouchon method of order 1. Alias of `Rouchon`.
      - `Rouchon1_5`: Rouchon method of order 1 with built-in Kraus map trace
      renormalization. Ideal for time-independent problems.
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
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`. Note:
            For fixed time step solvers, `t_save` does not define the time step.
            However, all `t_save` values should be aligned with the time step.
        exp_ops (Tensor, or list of Tensors, optional): List of operators for which the
            expectation value is computed at every time value in `t_save`.
        options (Options, optional): Solver options. See the list of available solvers.
        dtype (torch.dtype, optional): Complex data type to which all complex-valued
            tensors are converted. `t_save` is also converted to a real data type of
            the corresponding precision.
        device (torch.device, optional): Device on which the tensors are stored.

    Returns:
        A tuple `(rho_save, exp_save)` where
            `rho_save` is a tensor with the computed density matrices at `t_save`
                times, and of shape `(len(t_save), n, n)` or `(b_H, b_rho, len(t_save),
                n, n)` if batched. If `save_states` is `False`, only the final density
                matrix is returned with the same shape as the initial input.
            `exp_save` is a tensor with the computed expectation values at `t_save`
                times, and of shape `(len(exp_ops), len(t_save))` or `(b_H, b_rho,
                len(exp_ops), len(t_save))` if batched. `None` if no `exp_ops` are
                passed.
    """
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list of tensors. Otherwise,'
            ' consider using `sesolve`.'
        )

    # format all tensors for batching
    formatter = TensorFormatter(dtype, device)
    H_batched, rho0_batched = formatter.batch_H_and_state(H, rho0, state_to_dm=True)
    exp_ops = formatter.batch(exp_ops)
    jump_ops = formatter.batch(jump_ops)

    # convert t_save to a tensor
    t_save = torch.as_tensor(t_save, dtype=dtype_complex_to_real(dtype), device=device)

    # default options
    options = options or Dopri5()

    # define the solver
    args = (
        H_batched,
        rho0_batched,
        t_save,
        exp_ops,
        options,
    )
    if isinstance(options, Rouchon1):
        solver = MERouchon1(*args, jump_ops=jump_ops)
    elif isinstance(options, Rouchon1_5):
        solver = MERouchon1_5(*args, jump_ops=jump_ops)
    elif isinstance(options, Rouchon2):
        solver = MERouchon2(*args, jump_ops=jump_ops)
    elif isinstance(options, ODEAdaptiveStep):
        solver = MEAdaptive(*args, jump_ops=jump_ops)
    elif isinstance(options, Euler):
        solver = MEEuler(*args, jump_ops=jump_ops)
    else:
        raise NotImplementedError(f'Solver options {type(options)} is not implemented.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    rho_save, exp_save = solver.y_save, solver.exp_save
    rho_save = formatter.unbatch(rho_save)
    exp_save = formatter.unbatch(exp_save)

    return rho_save, exp_save
