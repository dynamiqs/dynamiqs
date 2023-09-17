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
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: str = 'dopri5',
    gradient: str | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the Schrödinger equation.

    Evolve the wavefunction $\ket{\psi(t)}$ from an initial state $\ket{\psi(t=0)} =
    \ket{\psi_0}$ according to the Schrödinger equation using a given Hamiltonian
    $H(t)$. The Schrödinger equation is given by

    $$
        \frac{d\ket{\psi}}{dt} = -i H \ket{\psi}.
    $$

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Extra Hamiltonian arguments are not yet
    supported.

    The Hamiltonian `H` and the initial wavefunction `psi0` can be batched over to
    solve multiple Schrödinger equations in a single run. The time list `tsave` is
    then common to all batches.

    `sesolve` can be differentiated through using either the default PyTorch autograd
    library (pass `gradient_alg="autograd"` in `options`). By default, if no
    gradient is required, the graph of operations is not stored to improve performance.

    Args:
        H _(Tensor or Callable)_: Hamiltonian.
            Can be a tensor of shape `(n, n)` or `(b_H, n, n)` if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=tsave[-1]`.
        psi0 _(Tensor)_: Initial wavefunction.
            Tensor of shape `(n, 1)` or `(b_rho, n, 1)` if batched.
        tsave _(Tensor, np.ndarray or list)_: Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=tsave[-1]`.
        exp_ops _(Tensor, or list of Tensors, optional)_: List of operators for which
            the expectation value is computed at every time value in `tsave`.
        solver _(str, optional)_: Solver to use. See the list of available solvers.
            Defaults to `"dopri5"`.
        gradient _(str, optional)_: Algorithm used for computing gradients.
            Can be either `"autograd"` or `None`. Defaults to `None`.
        options _(dict, optional)_: Solver options. See the list of available
            solvers, and the options common to all solver below.

    Note-: Available solvers
      - `dopri5` --- Dormand-Prince method of order 5 (adaptive step). Default solver.
      - `euler` --- Euler method (fixed step).
      - `propagator` --- Exact propagator computation through matrix exponentiation
        (fixed step). Only for time-independent problems.

    Note-: Available keys for `options`
        Common to all solvers:

        - **save_states** _(bool, optional)_ – If `True`, the state is saved at every
            time in `tsave`. If `False`, only the final state is stored and returned.
            Defaults to `True`.
        - **verbose** _(bool, optional)_ – If `True`, prints information about the
            integration progress. Defaults to `True`.
        - **dtype** _(torch.dtype, optional)_ – Complex data type to which all
            complex-valued tensors are converted. `tsave` is also converted to a real
            data type of the corresponding precision.
        - **device** _(torch.device, optional)_ – Device on which the tensors are
            stored.

        Required for fixed step solvers (`euler`, `propagator`):

        - **dt** _(float)_ – Numerical time step of integration.

        Optional for adaptive step solvers (`dopri5`):

        - **atol** _(float, optional)_ – Absolute tolerance. Defaults to `1e-12`.
        - **rtol** _(float, optional)_ – Relative tolerance. Defaults to `1e-6`.
        - **max_steps** _(int, optional)_ – Maximum number of steps. Defaults to `1e6`.
        - **safety_factor** _(float, optional)_ – Safety factor in the step size
            prediction. Defaults to `0.9`.
        - **min_factor** _(float, optional)_ – Minimum factor by which the step size can
            decrease in a single step. Defaults to `0.2`.
        - **max_factor** _(float, optional)_ – Maximum factor by which the step size can
            increase in a single step. Defaults to `10.0`.

    Warning: Warning for fixed step solvers
        For fixed time step solvers, the time list `tsave` should be strictly
        included in the time list used by the solver, given by `[0, dt, 2 * dt, ...]`
        where `dt` is defined with the `options` argument.

    Returns:
        Result of the master equation integration, as an instance of the `Result` class.
            The `result` object has the following attributes:

              - **ysave** or **states** _(Tensor)_ – Saved states.
              - **exp_save** or **expects** _(Tensor)_ – Saved expectation values.
              - **solver_str** (str): String representation of the solver.
              - **start_datetime** _(datetime)_ – Start time of the integration.
              - **end_datetime** _(datetime)_ – End time of the integration.
              - **total_time** _(datetime)_ – Total time of the integration.
              - **options** _(dict)_ – Solver options.
    """
    # H: (b_H?, n, n), psi0: (b_psi0?, n, 1) -> (ysave, exp_save) with
    #    - ysave: (b_H?, b_psi0?, len(tsave), n, 1)
    #    - exp_save: (b_H?, b_psi0?, len(exp_ops), len(tsave))

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
    result.ysave = result.ysave.squeeze(1).squeeze(0)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(1).squeeze(0)

    return result
