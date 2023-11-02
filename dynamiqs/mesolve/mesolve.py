from __future__ import annotations

from typing import Any

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

    Quote: Computing gradient
        `mesolve` can be differentiated through using PyTorch autograd library with
        `gradient="autograd"`, or the adjoint state method with `gradient="adjoint"`.

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
        solver _(str, optional)_: Solver for the differential equation integration (see
            the list below). Defaults to `"dopri5"`.
        gradient _(str, optional)_: Algorithm used to compute the gradient. Can be
            either `None`, `"autograd"` (PyTorch autograd library) or `"adjoint"`
            (differentiation with the adjoint state method). Defaults to `None`.
        options _(dict, optional)_: Solver options (see the list below). Defaults to
            `None`.

    Note-: Available solvers
      - `dopri5`: Dormand-Prince method of order 5 (adaptive step size ODE solver).
      - `euler`: Euler method (fixed step size ODE solver), not recommended
        except for testing purposes.
      - `rouchon1`: Rouchon method of order 1 (fixed step size ODE solver).
      - `rouchon2`: Rouchon method of order 2 (fixed step size ODE solver).
      - `propagator`: Explicitly compute the Liouvillian exponential to evolve
        the state between each time in `tsave`. Not recommended for systems of large
        dimension, due to the $\mathcal{O}(n^6)$ scaling of computing the Liouvillian
        exponential.

    Note-: Available options
        Common to all solvers:

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

        Required for fixed step size ODE solvers (`euler`, `rouchon1`, `rouchon2`):

        - **dt** _(float)_ – Numerical step size of integration.

        Optional for adaptive step size ODE solvers (`dopri5`):

        - **atol** _(float, optional)_ – Absolute tolerance. Defaults to `1e-8`.
        - **rtol** _(float, optional)_ – Relative tolerance. Defaults to `1e-6`.
        - **max_steps** _(int, optional)_ – Maximum number of steps. Defaults to `1e5`.
        - **safety_factor** _(float, optional)_ – Safety factor in the step size
            prediction. Defaults to `0.9`.
        - **min_factor** _(float, optional)_ – Minimum factor by which the step size can
            decrease in a single step. Defaults to `0.2`.
        - **max_factor** _(float, optional)_ – Maximum factor by which the step size can
            increase in a single step. Defaults to `5.0`.

        Optional for solver `rouchon1`:

        - **sqrt_normalization** _(bool, optional)_ – If `True`, the Kraus map is
            renormalized at every step to preserve the trace of the density matrix.
            Recommended only for time-independent problems. Ideal for stiff problems.
            Defaults to `False`.

        Required for `gradient="adjoint"`:

        - **parameters** _(tuple of nn.Parameter)_ – Parameters with respect to which
            the gradient is computed.

    Warning: Fixed step size ODE solvers
        For ODE solvers with fixed step sizes, the times in `tsave` must be multiples
        of the numerical step size of integration `dt` defined in the `options`
        argument.

    Returns:
        Object holding the results of the master equation integration. It has the
            following attributes:

            - **states** _(Tensor)_ – Saved states with shape
                _(bH?, brho?, len(tsave), n, n)_.
            - **expects** _(Tensor, optional)_ – Saved expectation values with shape
                _(bH?, brho?, len(exp_ops), len(tsave))_.
            - **tsave** or **times** _(Tensor)_ – Times for which states and expectation
                values were saved.
            - **solver_str** (str): Solver used.
            - **start_datetime** _(datetime)_ – Start date and time of the integration.
            - **end_datetime** _(datetime)_ – End date and time of the integration.
            - **total_time** _(timedelta)_ – Total duration of the integration.
            - **options** _(dict)_ – Solver options passed by the user.

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
    args = (H, rho0, tsave, exp_ops, options)
    solver = SOLVER_CLASS(*args, jump_ops=jump_ops)

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.ysave = result.ysave.squeeze(0, 1)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(0, 1)

    return result
