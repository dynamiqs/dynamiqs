from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..options import Dopri5, Euler, ODEAdaptiveStep, Options
from ..tensor_formatter import TensorFormatter
from ..utils.tensor_types import (
    OperatorLike,
    TDOperatorLike,
    TensorLike,
    dtype_complex_to_real,
)
from .adaptive import SEAdaptive
from .euler import SEEuler
from .options import Propagator
from .propagator import SEPropagator


def sesolve(
    H: TDOperatorLike,
    psi0: OperatorLike,
    t_save: TensorLike,
    *,
    exp_ops: OperatorLike | list[OperatorLike] | None = None,
    options: Options | None = None,
    gradient_alg: Literal['autograd', 'adjoint'] | None = None,
    parameters: tuple[nn.Parameter, ...] | None = None,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor | None]:
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
    # TODO add test to check that psi0 has the correct shape
    formatter = TensorFormatter(dtype, device)
    H_batched, psi0_batched = formatter.batch_H_and_state(H, psi0)
    exp_ops = formatter.batch(exp_ops)

    # convert t_save to tensor
    t_save = torch.as_tensor(t_save, dtype=dtype_complex_to_real(dtype), device=device)

    # default options
    options = options or Dopri5()

    # define the solver
    args = (
        H_batched,
        psi0_batched,
        t_save,
        exp_ops,
        options,
        gradient_alg,
        parameters,
    )
    if isinstance(options, Euler):
        solver = SEEuler(*args)
    elif isinstance(options, ODEAdaptiveStep):
        solver = SEAdaptive(*args)
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
