from __future__ import annotations

from typing import Any

import torch

from .._utils import check_time_tensor, obj_type_str
from ..gradient import Gradient
from ..solver import Dopri5, Euler, Propagator, Rouchon1, Rouchon2, Solver
from ..solvers.options import Options
from ..solvers.result import Result
from ..solvers.utils.td_tensor import to_td_tensor
from ..solvers.utils.utils import format_L
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from ..utils.utils import todm
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
        \frac{\dd\rho(t)}{\dt} =-i[H(t), \rho(t)]
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
    # === default solver
    if solver is None:
        solver = Dopri5()

    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
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

    # === check jump_ops
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

    # === check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but'
            f' has type {obj_type_str(exp_ops)}.'
        )

    # === convert and batch H, L, y0, exp_ops
    kw = dict(dtype=options.cdtype, device=options.device)

    # convert and batch H
    H = to_td_tensor(H, **kw)  # (bH?, n, n)
    n = H.size(-1)

    # convert and batch L
    L = [to_tensor(x, **kw) for x in jump_ops]  # [(??, n, n)]
    L = format_L(L)  # (nL, bL, n, n)
    nL = L.size(0)

    # convert and batch y0
    y0 = to_tensor(rho0, **kw)  # (by?, n, n)
    y0 = todm(y0)  # convert y0 to a density matrix

    if not options.flat_batching:
        H = H.view(-1, 1, 1, n, n)  # (bH, 1, 1, n, n) with bH = 1 if not batched
        bH = H.size(0)
        L = L.view(
            nL, 1, -1, 1, n, n
        )  # (nL, 1, bL, 1, n, n) with bL = 1 if not batched
        bL = L.size(2)

        y0 = y0.view(1, 1, -1, n, n)  # (1, 1, by, n, n) with by = 1 if not batched
        y0 = y0.repeat(bH, bL, 1, 1, 1)  # (bH, bL, by, n, n)
    else:
        if H.dim() == 3:
            bH = H.size(0)
        else:
            bH = 1
            H = H.view(-1, n, n)
        bL = L.size(1)  # (nL, bL, n, n) at this point
        if y0.dim() == 3:
            by = y0.size(0)
        else:
            by = 1
            y0 = y0.view(-1, n, n)

        if len({batch_dim for batch_dim in [bH, bL, by] if batch_dim > 1}) > 1:
            raise ValueError(
                f"Expected all batch dimensions the same or 1, got bH={bH}, bL={bL},"
                f" by={by}"
            )

        b = max(bH, bL, by)
        if by == 1:
            y0 = y0.repeat(b, 1, 1)

    # convert exp_ops
    exp_ops = to_tensor(exp_ops, **kw)  # (nE, n, n)

    # === convert tsave and init tmeas
    kw = dict(dtype=options.rdtype, device='cpu')
    tsave = to_tensor(tsave, **kw)
    check_time_tensor(tsave, arg_name='tsave')
    tmeas = torch.empty(0, **kw)

    # === define the solver
    solver = SOLVER_CLASS(H, y0, tsave, tmeas, exp_ops, options, L=L)

    # === compute the result
    result = solver.run()

    # === get saved tensors and restore initial batching
    if result.ysave is not None:
        if not options.flat_batching:
            result.ysave = result.ysave.squeeze(0, 1, 2)
        else:
            result.ysave = result.ysave.squeeze(0)
    if result.exp_save is not None:
        if not options.flat_batching:
            result.exp_save = result.exp_save.squeeze(0, 1, 2)
        else:
            result.exp_save = result.exp_save.squeeze(0)

    return result
