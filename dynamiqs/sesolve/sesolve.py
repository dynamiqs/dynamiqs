from __future__ import annotations

from .._utils import obj_type_str
from ..options import Dopri5, Euler, Options, Propagator
from ..solvers.result import Result
from ..solvers.utils.batching import batch_H, batch_y0
from ..solvers.utils.td_tensor import to_td_tensor
from ..solvers.utils.utils import check_time_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from .adaptive import SEDormandPrince5
from .euler import SEEuler
from .propagator import SEPropagator


def sesolve(
    H: TDArrayLike,
    psi0: ArrayLike,
    t_save: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
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
    # H: (b_H, 1, n, n)
    # psi0: (b_H, b_psi0, n, 1)
    # exp_ops: (len(exp_ops), n, n)
    H = to_td_tensor(H, dtype=options.cdtype, device=options.device)
    psi0 = to_tensor(psi0, dtype=options.cdtype, device=options.device)
    H = batch_H(H)
    psi0 = batch_y0(psi0, H)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)

    # convert t_save to tensor
    t_save = to_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # define the solver
    args = (H, psi0, t_save, exp_ops, options)
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
    result.y_save = result.y_save.squeeze(0, 1)
    result.exp_save = result.exp_save.squeeze(0, 1)

    return result
