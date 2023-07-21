from __future__ import annotations

import torch

from ..options import Dopri5, Euler, Options, Rouchon1, Rouchon2
from ..solvers.result import Result
from ..solvers.utils.tensor_formatter import TensorFormatter
from ..solvers.utils.utils import check_time_tensor
from ..utils.tensor_types import OperatorLike, TDOperatorLike, TensorLike
from .adaptive import MEDormandPrince5
from .euler import MEEuler
from .rouchon import MERouchon1, MERouchon2


def mesolve(
    H: TDOperatorLike,
    jump_ops: list[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: list[OperatorLike] | None = None,
    options: Options | None = None,
) -> Result:
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
      - `Rouchon1`: Rouchon method of order 1. Alias of `Rouchon`. Set the
        `sqrt_normalization` option to `True` to enable built-in Kraus map trace
        renormalization, ideal for time-independent and/or stiff problems.
      - `Rouchon2`: Rouchon method of order 2.
      - `Euler`: Euler method.

    Args:
        H _(Tensor or Callable)_: Hamiltonian.
            Can be a tensor of shape `(n, n)` or `(b_H, n, n)` if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=t_save[-1]`.
        jump_ops _(Tensor, or list of Tensors)_: List of jump operators.
            Each jump operator should be a tensor of shape `(n, n)`.
        rho0 _(Tensor)_: Initial density matrix.
            Tensor of shape `(n, n)` or `(b_rho, n, n)` if batched.
        t_save _(Tensor, np.ndarray or list)_: Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`. Note:
            For fixed time step solvers, `t_save` does not define the time step.
            However, all `t_save` values should be aligned with the time step.
        exp_ops _(Tensor, or list of Tensors, optional)_: List of operators for which
            the expectation value is computed at every time value in `t_save`.
        options _(Options, optional)_: Solver options. See the list of available
            solvers.

    Returns:
        Result of the master equation integration.
    """
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (y_save, exp_save) with
    #    - y_save: (b_H?, b_rho0?, len(t_save), n, n)
    #    - exp_save: (b_H?, b_rho0?, len(exp_ops), len(t_save))

    # default options
    options = options or Dopri5()

    if isinstance(jump_ops, list) and len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list of tensors. Otherwise,'
            ' consider using `sesolve`.'
        )

    # format and batch all tensors
    formatter = TensorFormatter(options.cdtype, options.device)
    H_batched, rho0_batched = formatter.format_H_and_state(H, rho0, state_to_dm=True)
    # H_batched: (b_H, 1, n, n)
    # rho0_batched: (b_H, b_rho0, n, n)
    exp_ops = formatter.format(exp_ops)  # (len(exp_ops), n, n)
    jump_ops = formatter.format(jump_ops)  # (len(jump_ops), n, n)

    # convert t_save to a tensor
    t_save = torch.as_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # define the solver
    args = (H_batched, rho0_batched, t_save, exp_ops, options)
    if isinstance(options, Rouchon1):
        solver = MERouchon1(*args, jump_ops=jump_ops)
    elif isinstance(options, Rouchon2):
        solver = MERouchon2(*args, jump_ops=jump_ops)
    elif isinstance(options, Dopri5):
        solver = MEDormandPrince5(*args, jump_ops=jump_ops)
    elif isinstance(options, Euler):
        solver = MEEuler(*args, jump_ops=jump_ops)
    else:
        raise NotImplementedError(f'Solver options {type(options)} is not implemented.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = formatter.unbatch(result.y_save)
    result.exp_save = formatter.unbatch(result.exp_save)

    return result
