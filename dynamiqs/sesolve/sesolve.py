from __future__ import annotations

from typing import Any

from .._utils import obj_type_str
from ..solvers.options import Dopri5, Euler, Propagator
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, check_time_tensor, to_td_tensor
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
    solver: str = 'dopri5',
    gradient: str | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    """Solve the Schrödinger equation."""
    # H: (b_H?, n, n), psi0: (b_psi0?, n, 1) -> (y_save, exp_save) with
    #    - y_save: (b_H?, b_psi0?, len(t_save), n, 1)
    #    - exp_save: (b_H?, b_psi0?, len(exp_ops), len(t_save))

    # TODO support density matrices too
    # TODO add test to check that psi0 has the correct shape

    # options
    if options is None:
        options = {}
    options['gradient_alg'] = gradient
    if solver == 'dopri5':
        options = Dopri5(**options)
        SOLVER_CLASS = SEDormandPrince5
    elif solver == 'euler':
        options = Euler(**options)
        SOLVER_CLASS = SEEuler
    elif solver == 'propagator':
        options = Propagator(**options)
        SOLVER_CLASS = SEPropagator
    else:
        raise ValueError(f'Solver "{solver}" is not supported.')

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
    H = to_td_tensor(
        H, dtype=options.cdtype, device=options.device, args=options.H_args
    )
    psi0 = to_tensor(psi0, dtype=options.cdtype, device=options.device)
    H = batch_H(H)
    psi0 = batch_y0(psi0, H)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)

    # convert t_save to tensor
    t_save = to_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # define the solver
    args = (H, psi0, t_save, exp_ops, options)
    solver = SOLVER_CLASS(*args)

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = result.y_save.squeeze(1).squeeze(0)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(1).squeeze(0)

    return result
