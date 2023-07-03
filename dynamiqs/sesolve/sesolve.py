from __future__ import annotations

import torch
from torch import Tensor

from ..options import Dopri5, Euler, Options, Propagator
from ..utils.solver_utils import check_time_array
from ..utils.tensor_formatter import TensorFormatter
from ..utils.tensor_types import OperatorLike, TDOperatorLike, TensorLike
from .adaptive import SEDormandPrince5
from .euler import SEEuler
from .propagator import SEPropagator


def sesolve(
    H: TDOperatorLike,
    psi0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: OperatorLike | list[OperatorLike] | None = None,
    options: Options | None = None,
) -> tuple[Tensor, Tensor | None]:
    # H: (b_H?, n, n), psi0: (b_psi0?, n, 1) -> (y_save, exp_save) with
    #    - y_save: (b_H?, b_psi0?, len(t_save), n, 1)
    #    - exp_save: (b_H?, b_psi0?, len(exp_ops), len(t_save))

    # TODO support density matrices too
    # TODO add test to check that psi0 has the correct shape

    # format and batch all tensors
    formatter = TensorFormatter(options.dtype, options.device)
    H_batched, psi0_batched = formatter.format_H_and_state(H, psi0)
    # H_batched: (b_H, 1, n, n)
    # psi0_batched: (b_H, b_psi0, n, 1)
    exp_ops = formatter.format(exp_ops)  # (len(exp_ops), n, n)

    # convert t_save to tensor
    t_save = torch.as_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_array(t_save, 't_save')

    # default options
    options = options or Dopri5()

    # define the solver
    t_save_all = t_save
    args = (H_batched, psi0_batched, t_save_all, t_save, exp_ops, options)
    if isinstance(options, Euler):
        solver = SEEuler(*args)
    elif isinstance(options, Dopri5):
        solver = SEDormandPrince5(*args)
    elif isinstance(options, Propagator):
        solver = SEPropagator(*args)
    else:
        raise NotImplementedError(f'Solver options {type(options)} is not implemented.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    psi_save, exp_save = solver.y_save, solver.exp_save
    psi_save = formatter.unbatch(psi_save)
    exp_save = formatter.unbatch(exp_save)

    return psi_save, exp_save
