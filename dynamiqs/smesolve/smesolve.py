from __future__ import annotations

import torch

from .._utils import obj_type_str
from ..options import Euler, Options, Rouchon1SME
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, check_time_tensor, to_td_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from ..utils.utils import is_ket, ket_to_dm
from .euler import SMEEuler
from .rouchon import SMERouchon1


def smesolve(
    H: TDArrayLike,
    jump_ops: list[ArrayLike],
    rho0: ArrayLike,
    t_save: ArrayLike,
    etas: ArrayLike,
    ntrajs: int,
    *,
    t_meas: ArrayLike | None = None,
    seed: int | None = None,
    exp_ops: list[ArrayLike] | None = None,
    options: Options | None = None,
) -> Result:
    """Solve the Stochastic master equation."""
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (y_save, exp_save, meas_save) with
    #    - y_save: (b_H?, b_rho0?, ntrajs, len(t_save), n, n)
    #    - exp_save: (b_H?, b_rho0?, ntrajs, len(exp_ops), len(t_save))
    #    - meas_save: (b_H?, b_rho0?, ntrajs, len(meas_ops), len(t_meas) - 1)

    # default options
    if options is None:
        raise ValueError(
            'No default solver yet, please specify one using the `options` argument.'
        )

    # check jump_ops
    if not isinstance(jump_ops, list):
        raise TypeError(
            'Argument `jump_ops` must be a list of array-like objects, but has type'
            f' {obj_type_str(jump_ops)}.'
        )
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list, otherwise consider using'
            ' `ssesolve`.'
        )

    # check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but'
            f' has type {obj_type_str(exp_ops)}.'
        )

    # format and batch all tensors
    # H: (b_H, 1, 1, n, n)
    # rho0: (b_H, b_rho0, ntrajs, n, n)
    # exp_ops: (len(exp_ops), n, n)
    # jump_ops: (len(jump_ops), n, n)
    H = to_td_tensor(H, dtype=options.cdtype, device=options.device)
    rho0 = to_tensor(rho0, dtype=options.cdtype, device=options.device)
    H = batch_H(H).unsqueeze(2)
    rho0 = batch_y0(rho0, H).unsqueeze(2).repeat(1, 1, ntrajs, 1, 1)
    if is_ket(rho0):
        rho0 = ket_to_dm(rho0)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)
    jump_ops = to_tensor(jump_ops, dtype=options.cdtype, device=options.device)

    # convert t_save to a tensor
    t_save = to_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # convert etas to a tensor and check
    etas = to_tensor(etas, dtype=options.rdtype, device=options.device)
    if len(etas) != len(jump_ops):
        raise ValueError(
            'Argument `etas` must have the same length as `jump_ops` of length'
            f' {len(jump_ops)}, but has length {len(etas)}.'
        )
    if torch.all(etas == 0.0):
        raise ValueError(
            'Argument `etas` must contain at least one non-zero value, otherwise '
            'consider using `mesolve`.'
        )
    if torch.any(etas < 0.0) or torch.any(etas > 1.0):
        raise ValueError('Argument `etas` must contain values between 0 and 1.')

    # convert t_meas to a tensor
    t_meas = to_tensor(t_meas, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_meas, arg_name='t_meas', allow_empty=True)

    # define random number generator from seed
    generator = torch.Generator(device=options.device)
    generator.seed() if seed is None else generator.manual_seed(seed)

    # define the solver
    args = (H, rho0, t_save, exp_ops, options)
    kwargs = dict(
        jump_ops=jump_ops,
        etas=etas,
        generator=generator,
        t_meas=t_meas,
    )
    if isinstance(options, Rouchon1SME):
        solver = SMERouchon1(*args, **kwargs)
    elif isinstance(options, Euler):
        solver = SMEEuler(*args, **kwargs)
    else:
        raise ValueError(f'Solver options {obj_type_str(options)} is not supported.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = result.y_save.squeeze(1).squeeze(0)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(1).squeeze(0)
    if result.meas_save is not None:
        result.meas_save = result.meas_save.squeeze(1).squeeze(0)

    return result
