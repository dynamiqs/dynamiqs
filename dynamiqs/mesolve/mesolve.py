from __future__ import annotations

from typing import Any

import torch

from .._utils import check_time_tensor, obj_type_str
from ..gradient import Gradient
from ..solver import Dopri5, Euler, Propagator, Rouchon1, Rouchon2, Solver
from ..solvers.options import Options
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, to_td_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from ..utils.utils import isket, todm
from .adaptive import MEDormandPrince5
from .euler import MEEuler
from .propagator import MEPropagator
from .rouchon import MERouchon1, MERouchon2


def mesolve(
    H: TDArrayLike,
    jump_ops: list[ArrayLike],
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the Lindblad master equation.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho(t=0)$, according to the Lindblad master
    equation (with $\hbar=1$):
    $$
        \frac{\mathrm{d}\rho(t)}{\mathrm{d}t} =-i[H(t), \rho(t)]
        + \sum_{k=1}^N \left(
            L_k \rho(t) L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho(t)
            - \frac{1}{2} \rho(t) L_k^\dag L_k
        \right),
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k\}$ is a collection
    of jump operators.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be passed as a function with
        signature `H(t: float) -> Tensor`.

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial density matrix `rho0` can be batched to
        solve multiple master equations concurrently. All other arguments are common
        to every batch.

    Args:
        H _(array-like or function, shape (n, n) or (bH, n, n))_: Hamiltonian. For
            time-dependent problems, provide a function with signature
            `H(t: float) -> Tensor` that returns a tensor (batched or not) for any
            given time between `t = 0.0` and `t = tsave[-1]`.
        jump_ops _(list of 2d array-like, each with shape (n, n))_: List of jump
            operators.
        rho0 _(array-like, shape (n, n) or (brho, n, n))_: Initial density matrix.
        tsave _(1d array-like)_: Times at which the states and expectation values are
            saved. The master equation is solved from `t = 0.0` to `t = tsave[-1]`.
        exp_ops _(list of 2d array-like, each with shape (n, n), optional)_: List of
            operators for which the expectation value is computed. Defaults to `None`.
        solver _(Solver, optional)_: Solver for the differential equation integration
            (see the list below). Defaults to `dq.solver.Dopri5()`.
        gradient _(Gradient, optional)_: Algorithm used to compute the gradient (see
            the list below). Defaults to `None`.
        options _(dict, optional)_: Generic options (see the list below). Defaults to
            `None`.

    Note-: Available solvers
      - `dq.solver.Dopri5`: Dormand-Prince method of order 5 (adaptive step size ODE
         solver).
      - `dq.solver.Euler`: Euler method (fixed step size ODE solver), not recommended
         except for testing purposes.
      - `dq.solver.Rouchon1`: Rouchon method of order 1 (fixed step size ODE solver).
      - `dq.solver.Rouchon2`: Rouchon method of order 2 (fixed step size ODE solver).
      - `dq.solver.Propagator`: Explicitly compute the Liouvillian exponential to evolve
         the state between each time in `tsave`. Not recommended for systems of large
         dimension, due to the $\mathcal{O}(n^6)$ scaling of computing the Liouvillian
         exponential.

    Note-: Available gradient algorithms
        - `None`: No gradient.
        - `dq.gradient.Autograd`: PyTorch autograd library
        - `dq.gradient.Adjoint`: Differentiation with the adjoint state method.

    Note-: Available options
        - **save_states** _(bool, optional)_ – If `True`, the state is saved at every
            time in `tsave`. If `False`, only the final state is returned. Defaults to
            `True`.
        - **verbose** _(bool, optional)_ – If `True`, print a progress bar during the
            integration. Defaults to `True`.
        - **dtype** _(torch.dtype, optional)_ – Complex data type to which all
            complex-valued tensors are converted. `tsave` is converted to a real data
            type of the corresponding precision. Defaults to the complex data type set
            by `torch.set_default_dtype`.
        - **device** _(torch.device, optional)_ – Device on which the tensors are
            stored. Defaults to the device set by `torch.set_default_device`.

    Returns:
        Object holding the results of the master equation integration. It has the
            following attributes:

            - **states** _(Tensor)_ – Saved states with shape
                _(bH?, brho?, len(tsave), n, n)_.
            - **expects** _(Tensor, optional)_ – Saved expectation values with shape
                _(bH?, brho?, len(exp_ops), len(tsave))_.
            - **tsave** or **times** _(Tensor)_ – Times for which states and expectation
                values were saved.
            - **start_datetime** _(datetime)_ – Start date and time of the integration.
            - **end_datetime** _(datetime)_ – End date and time of the integration.
            - **total_time** _(timedelta)_ – Total duration of the integration.
            - **solver** (Solver) –  Solver used.
            - **gradient** (Gradient) – Gradient used.
            - **options** _(dict)_  – Options used.

    Warning: Time-dependent jump operators
        Time-dependent jump operators are not yet supported. If this is a required
        feature, we would be glad to discuss it, please
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).
    """

    # default solver
    if solver is None:
        solver = Dopri5()

    # options
    options = Options(solver=solver, gradient=gradient, options=options)

    # solver class
    solvers = {
        Propagator: MEPropagator,
        Euler: MEEuler,
        Rouchon1: MERouchon1,
        Rouchon2: MERouchon2,
        Dopri5: MEDormandPrince5,
    }
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    SOLVER_CLASS = solvers[type(solver)]

    # check jump_ops
    if not isinstance(jump_ops, list):
        raise TypeError(
            'Argument `jump_ops` must be a list of array-like objects, but has type'
            f' {obj_type_str(jump_ops)}.'
        )
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list, otherwise consider using'
            ' `sesolve`.'
        )
    # check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but'
            f' has type {obj_type_str(exp_ops)}.'
        )

    # format and batch all tensors
    # H: (b_H, 1, n, n)
    # rho0: (b_H, b_rho0, n, n)
    # exp_ops: (len(exp_ops), n, n)
    # jump_ops: (len(jump_ops), n, n)
    H = to_td_tensor(H, dtype=options.cdtype, device=options.device)
    rho0 = to_tensor(rho0, dtype=options.cdtype, device=options.device)
    H = batch_H(H)
    rho0 = batch_y0(rho0, H)
    if isket(rho0):
        rho0 = todm(rho0)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)
    jump_ops = to_tensor(jump_ops, dtype=options.cdtype, device=options.device)

    # convert tsave to a tensor
    tsave = to_tensor(tsave, dtype=options.rdtype, device=options.device)
    check_time_tensor(tsave, arg_name='tsave')

    # define the solver
    tmeas = torch.empty(0)
    solver = SOLVER_CLASS(H, rho0, tsave, tmeas, exp_ops, options, jump_ops=jump_ops)

    # compute the result
    result = solver.run()

    # get saved tensors and restore correct batching
    result.ysave = result.ysave.squeeze(0, 1)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(0, 1)

    return result
