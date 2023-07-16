from __future__ import annotations

import torch

from ..options import Dopri5, Euler, Options, Propagator
from ..solvers.result import Result
from ..solvers.utils.tensor_formatter import TensorFormatter
from ..solvers.utils.utils import check_time_tensor
from ..utils.tensor_types import OperatorLike, TDOperatorLike, TensorLike
from ..utils.utils import obj_type_str
from .adaptive import SEDormandPrince5
from .euler import SEEuler
from .propagator import SEPropagator


def sesolve(
    H: TDOperatorLike,
    psi0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: list[OperatorLike] | None = None,
    options: Options | None = None,
) -> Result:
    """Solve the Schrödinger equation."""
    # H: (b_H?, n, n), psi0: (b_psi0?, n, 1) -> (y_save, exp_save) with
    #    - y_save: (b_H?, b_psi0?, len(t_save), n, 1)
    #    - exp_save: (b_H?, b_psi0?, len(exp_ops), len(t_save))

    # TODO support density matrices too
    # TODO add test to check that psi0 has the correct shape

    # default options
    options = options or Dopri5()

    # check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but has'
            f' type {obj_type_str(exp_ops)}.'
        )

    # format and batch all tensors
    formatter = TensorFormatter(options.cdtype, options.device)
    H_batched, psi0_batched = formatter.format_H_and_state(H, psi0)
    # H_batched: (b_H, 1, n, n)
    # psi0_batched: (b_H, b_psi0, n, 1)
    exp_ops = formatter.format(exp_ops)  # (len(exp_ops), n, n)

    # convert t_save to tensor
    t_save = torch.as_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # define the solver
    args = (H_batched, psi0_batched, t_save, exp_ops, options)
    if isinstance(options, Euler):
        solver = SEEuler(*args)
    elif isinstance(options, Dopri5):
        solver = SEDormandPrince5(*args)
    elif isinstance(options, Propagator):
        solver = SEPropagator(*args)
    else:
        raise ValueError(f'Solver options {obj_type_str(options)} is not supported.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = formatter.unbatch(result.y_save)
    result.exp_save = formatter.unbatch(result.exp_save)

    return result
