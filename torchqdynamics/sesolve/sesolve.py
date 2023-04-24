from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..options import Dopri45, Euler, ODEAdaptiveStep, Options
from ..tensor_types import (
    OperatorLike,
    TDOperatorLike,
    TensorLike,
    dtype_complex_to_float,
    to_tensor,
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
    H = to_tensor(H, dtype=dtype, device=device, is_complex=True)
    H_batched = H[None, ...] if H.ndim == 2 else H

    # convert psi0 to a tensor and batch by default
    # TODO add test to check that psi0 has the correct shape
    b_H = H_batched.size(0)
    psi0 = to_tensor(psi0, dtype=dtype, device=device, is_complex=True)
    psi0_batched = psi0[None, ...] if psi0.ndim == 2 else psi0
    psi0_batched = psi0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_psi0, n, 1)

    # convert t_save to tensor
    t_save = torch.as_tensor(t_save, dtype=dtype_complex_to_float(dtype), device=device)

    # convert exp_ops to tensor
    exp_ops = to_tensor(exp_ops, dtype=dtype, device=device, is_complex=True)
    exp_ops = exp_ops[None, ...] if exp_ops.ndim == 2 else exp_ops

    # default options
    if options is None:
        options = Dopri45()

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
    if psi0.ndim == 2:
        psi_save = psi_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.ndim == 2:
        psi_save = psi_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return psi_save, exp_save
