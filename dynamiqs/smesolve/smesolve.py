from __future__ import annotations

from typing import Any

import torch

from .._utils import obj_type_str
from ..solvers.options import Euler, Rouchon1
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, check_time_tensor, to_td_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from ..utils.utils import isket, todm
from .euler import SMEEuler
from .rouchon import SMERouchon1


def smesolve(
    H: TDArrayLike,
    jump_ops: list[ArrayLike],
    etas: ArrayLike,
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    tmeas: ArrayLike | None = None,
    ntrajs: int = 1,
    seed: int | None = None,
    solver: str = '',
    gradient: str | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the stochastic master equation.

    Evolve the density matrix $\rho(t)$ from an initial state $\rho(t=0) = \rho_0$
    according to the stochastic master equation using a given Hamiltonian $H(t)$, a
    list of jump operators $\{L_k\}$ with respective measurement efficiencies
    $\{\eta_k\}$. The stochastic master equation is given by

    $$
    \begin{split}
        d\rho_t = -i[H, \rho_t] dt &+ \sum_k \left(L_k \rho_t L_k^\dagger -
        \frac{1}{2} \left\{L_k^\dagger L_k, \rho_t\right\}\right) dt \\
        &+ \sum_k \sqrt{\eta_k} \left( L_k \rho_t + \rho_t L_k^\dagger
        - \mathrm{Tr}[(L_k+L_k^\dagger) \rho_t] \right) dW_t^{(k)}
    \end{split}
    $$

    where $dW_t^{(k)} \sim N(0, t)$ is a Wiener process, numerically sampled from a
    white noise distribution. The measurement signals are then given by

    $$
        dy_t^{(k)} = \sqrt{\eta_k} \mathrm{Tr}[(L_k + L_k^\dagger) \rho_t] dt
        + dW_t^{(k)}.
    $$

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Extra Hamiltonian arguments and time-dependence
    for the jump operators are not yet supported.

    The Hamiltonian `H` and the initial density matrix `rho0` can be batched over to
    solve multiple stochastic master equations in a single run. The jump operators
    `jump_ops` and time list `tsave` are then common to all batches.

    `smesolve` can be differentiated through using either the default PyTorch autograd
    library (pass `gradient_alg="autograd"` in `options`), or a custom adjoint state
    differentiation (pass `gradient_alg="adjoint"` in `options`). By default, if no
    gradient is required, the graph of operations is not stored to improve performance.

    Args:
        H _(Tensor or Callable)_: Hamiltonian.
            Can be a tensor of shape `(n, n)` or `(b_H, n, n)` if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=tsave[-1]`.
        jump_ops _(Tensor, or list of Tensors)_: List of jump operators.
            Each jump operator should be a tensor of shape `(n, n)`.
        etas _(Tensor, np.ndarray or list)_: Measurement efficiencies, of same length
            as `jump_ops`.
        rho0 _(Tensor)_: Initial density matrix.
            Tensor of shape `(n, n)` or `(b_rho, n, n)` if batched.
        tsave _(Tensor, np.ndarray or list)_: Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=tsave[-1]`.
        exp_ops _(Tensor, or list of Tensors, optional)_: List of operators for which
            the expectation value is computed at every time value in `tsave`.
        tmeas _(Tensor, np.ndarray or list, optional)_: Times for which measurement
            signals are saved. Defaults to `tsave`.
        ntrajs _(int, optional)_: Number of stochastic trajectories. Defaults to 1.
        seed _(int, optional)_: Seed for the random number generator. Defaults to
            `None`.
        solver _(str, optional)_: Solver to use. See the list of available solvers.
            Defaults to `""` (no default solver).
        gradient _(str, optional)_: Algorithm used for computing gradients.
            Can be either `"autograd"` or `None`. Defaults to `None`.
        options _(dict, optional)_: Solver options. See the list of available
            solvers, and the options common to all solver below.

    Note-: Available solvers
      - `euler` --- Euler method (fixed step).
      - `rouchon1` --- Rouchon method of order 1 (fixed step).

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

        Required for fixed step solvers (`euler`, `rouchon1`):

        - **dt** _(float)_ – Numerical time step of integration.

    Warning: Warning for fixed step solvers
        For fixed time step solvers, both time lists `tsave` and `tmeas` should be
        strictly included in the time list used by the solver, given by
        `[0, dt, 2 * dt, ...]` where `dt` is defined with the `options` argument.

    Returns:
        Result of the master equation integration, as an instance of the `Result` class.
            The `result` object has the following attributes:

              - **ysave** or **states** _(Tensor)_ – Saved states.
              - **exp_save** or **expects** _(Tensor)_ – Saved expectation values.
              - **measurements** _(Tensor)_ – Saved measurement signals.
              - **solver_str** (str): String representation of the solver.
              - **start_datetime** _(datetime)_ – Start time of the integration.
              - **end_datetime** _(datetime)_ – End time of the integration.
              - **total_time** _(datetime)_ – Total time of the integration.
              - **options** _(dict)_ – Solver options.
    """
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (ysave, exp_save, meas_save) with
    #    - ysave: (b_H?, b_rho0?, ntrajs, len(tsave), n, n)
    #    - exp_save: (b_H?, b_rho0?, ntrajs, len(exp_ops), len(tsave))
    #    - meas_save: (b_H?, b_rho0?, ntrajs, len(meas_ops), len(tmeas) - 1)

    # default solver
    if solver == '':
        raise ValueError(
            'No default solver yet, please specify one using the `solver` argument.'
        )
    # options
    if options is None:
        options = {}
    options['gradient_alg'] = gradient
    if solver == 'euler':
        options = Euler(**options)
        SOLVER_CLASS = SMEEuler
    elif solver == 'rouchon1':
        options = Rouchon1(**options)
        SOLVER_CLASS = SMERouchon1
    else:
        raise ValueError(f'Solver "{solver}" is not supported.')

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
    if isket(rho0):
        rho0 = todm(rho0)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)
    jump_ops = to_tensor(jump_ops, dtype=options.cdtype, device=options.device)

    # convert tsave to a tensor
    tsave = to_tensor(tsave, dtype=options.rdtype, device=options.device)
    check_time_tensor(tsave, arg_name='tsave')

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

    # convert tmeas to a tensor (default to `tsave` if None)
    if tmeas is None:
        tmeas = tsave
    tmeas = to_tensor(tmeas, dtype=options.rdtype, device=options.device)
    check_time_tensor(tmeas, arg_name='tmeas', allow_empty=True)

    # define random number generator from seed
    generator = torch.Generator(device=options.device)
    generator.seed() if seed is None else generator.manual_seed(seed)

    # define the solver
    args = (H, rho0, tsave, exp_ops, options)
    kwargs = dict(
        jump_ops=jump_ops,
        etas=etas,
        generator=generator,
        tmeas=tmeas,
    )
    solver = SOLVER_CLASS(*args, **kwargs)

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.ysave = result.ysave.squeeze(1).squeeze(0)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(1).squeeze(0)
    if result.meas_save is not None:
        result.meas_save = result.meas_save.squeeze(1).squeeze(0)

    return result
