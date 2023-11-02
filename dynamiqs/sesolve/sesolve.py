from __future__ import annotations

from typing import Any

from .._utils import check_time_tensor, obj_type_str
from ..gradient import Gradient
from ..solver import Dopri5, Euler, Propagator, Solver
from ..solvers.options import Options
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, to_td_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from .adaptive import SEDormandPrince5
from .euler import SEEuler
from .propagator import SEPropagator


def sesolve(
    H: TDArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the Schrödinger equation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi(t=0)}$, according to the Schrödinger
    equation (with $\hbar=1$):
    $$
        \frac{\mathrm{d}\ket{\psi(t)}}{\mathrm{d}t} = -i H(t) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be passed as a function with
        signature `H(t: float) -> Tensor`.

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple Schrödinger equations concurrently. All other arguments are
        common to every batch.

    Quote: Computing gradient
        `sesolve` can be differentiated through using PyTorch autograd library with
        `gradient="autograd"`, or the adjoint state method with `gradient="adjoint"`.

    Args:
        H _(array-like or function, shape (n, n) or (bH, n, n))_: Hamiltonian. For
            time-dependent problems, provide a function with signature
            `H(t: float) -> Tensor` that returns a tensor (batched or not) for any
            given time between `t = 0.0` and `t = tsave[-1]`.
        psi0 _(array-like, shape (n, 1) or (bpsi, n, 1))_: Initial state vector.
        tsave _(1d array-like)_: Times at which the states and expectation values are
            saved. The Schrödinger equation is solved from `t = 0.0` to `t = tsave[-1]`.
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
      - `propagator`: Explicitly compute the propagator exponential to evolve the state
         between each time in `tsave`.

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

        Required for fixed step size ODE solvers (`euler`):

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

        Required for `gradient="adjoint"`:

        - **parameters** _(tuple of nn.Parameter)_ – Parameters with respect to which
            the gradient is computed.

    Warning: Fixed step size ODE solvers
        For ODE solvers with fixed step sizes, the times in `tsave` must be multiples
        of the numerical step size of integration `dt` defined in the `options`
        argument.

    Returns:
        Object holding the results of the Schrödinger equation integration. It has the
            following attributes:

            - **states** _(Tensor)_ – Saved states with shape
                _(bH?, bpsi?, len(tsave), n, 1)_.
            - **expects** _(Tensor, optional)_ – Saved expectation values with shape
                _(bH?, bpsi?, len(exp_ops), len(tsave))_.
            - **tsave** or **times** _(Tensor)_ – Times for which states and expectation
                values were saved.
            - **solver_str** (str): Solver used.
            - **start_datetime** _(datetime)_ – Start date and time of the integration.
            - **end_datetime** _(datetime)_ – End date and time of the integration.
            - **total_time** _(timedelta)_ – Total duration of the integration.
            - **options** _(dict)_ – Solver options passed by the user.
    """
    # H: (b_H?, n, n), psi0: (b_psi0?, n, 1) -> (ysave, exp_save) with
    #    - ysave: (b_H?, b_psi0?, len(tsave), n, 1)
    #    - exp_save: (b_H?, b_psi0?, len(exp_ops), len(tsave))

    # TODO support density matrices too
    # TODO add test to check that psi0 has the correct shape

    # default solver
    if solver is None:
        solver = Dopri5()

    # options
    options = Options(solver=solver, gradient=gradient, options=options)

    # solver class
    solvers = {
        Propagator: SEPropagator,
        Euler: SEEuler,
        Dopri5: SEDormandPrince5,
    }
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    SOLVER_CLASS = solvers[type(solver)]

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

    # convert tsave to tensor
    tsave = to_tensor(tsave, dtype=options.rdtype, device=options.device)
    check_time_tensor(tsave, arg_name='tsave')

    # define the solver
    args = (H, psi0, tsave, exp_ops, options)
    solver = SOLVER_CLASS(*args)

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.ysave = result.ysave.squeeze(0, 1)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(0, 1)

    return result
