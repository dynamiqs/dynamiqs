from __future__ import annotations

from typing import Any

import torch

from .._utils import check_time_tensor, obj_type_str
from ..gradient import Gradient
from ..solver import Euler, Rouchon1, Solver
from ..solvers.options import Options
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, to_td_tensor
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
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the diffusive stochastic master equation (SME).

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho(t=0)$, according to the diffusive SME in Itô
    form (with $\hbar=1$):
    $$
        \begin{split}
            \dd\rho(t) =&~ -i[H(t), \rho(t)] \dt \\\\
            &+ \sum_{k=1}^N \left(
                L_k \rho(t) L_k^\dag
                - \frac{1}{2} L_k^\dag L_k \rho(t)
                - \frac{1}{2} \rho(t) L_k^\dag L_k
            \right)\dt \\\\
            &+ \sum_{k=1}^N \sqrt{\eta_k} \left(
                L_k \rho(t)
                + \rho(t) L_k^\dag
                - \tr{(L_k+L_k^\dag)\rho(t)}\rho(t)\ \dd W_k(t)
            \right),
        \end{split}
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$, $\{L_k\}$ is a collection
    of jump operators, each continuously measured with efficiency $0\leq\eta_k\leq1$
    ($\eta_k=0$ for purely dissipative loss channels) and $\dd W_k(t)$ are
    independent Wiener processes.

    Notes:
        In quantum optics the diffusive SME corresponds to homodyne or heterodyne
        detection schemes, as opposed to the jump SME which corresponds to photon
        counting schemes. No solver for the jump SME is provided yet, if it is needed
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    The measured signals $I_k(t)=\dd y_k(t)/\dt$ verify:
    $$
        \dd y_k(t) =\sqrt{\eta_k} \tr{(L_k + L_k^\dag) \rho(t)} \dt + \dd W_k(t).
    $$

    Notes:
        Sometimes the signals are defined with a different but equivalent normalisation
        $\dd y_k'(t) = \dd y_k(t)/(2\sqrt{\eta_k})$.

    The signals $I_k(t)$ are singular quantities, the solver returns the averaged signals
    $J_k(t)$ defined for a time interval $[t_0, t_1]$ by:
    $$
        J_k([t_0, t_1]) = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} I_k(t) \dt
        = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} \dd y_k(t).
    $$
    The time intervals for integration are defined by the argument `t_meas`, which
    defines `len(t_meas) - 1` intervals. By default, `t_meas = tsave`, so the signals
    are averaged between the times at which the states are saved.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be passed as a function with
        signature `H(t: float) -> Tensor`.

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial density matrix `rho0` can be batched to
        solve multiple SMEs concurrently. All other arguments are common to every batch.

    Args:
        H _(array-like or function, shape (n, n) or (bH, n, n))_: Hamiltonian. For
            time-dependent problems, provide a function with signature
            `H(t: float) -> Tensor` that returns a tensor (batched or not) for any
            given time between `t = 0.0` and `t = tsave[-1]`.
        jump_ops _(list of 2d array-like, each with shape (n, n))_: List of jump
            operators.
        etas _(1d array-like)_: Measurement efficiencies, must be of the same length
            as `jump_ops` with values between 0 and 1. For a purely dissipative loss
            channel, set the corresponding efficiency to 0. No measurement signal will
            be returned for such channels.
        rho0 _(array-like, shape (n, n) or (brho, n, n))_: Initial density matrix.
        tsave _(1d array-like)_: Times at which the states and expectation values are
            saved. The SME is solved from `t = 0.0` to `t = tsave[-1]`.
        exp_ops _(list of 2d array-like, each with shape (n, n), optional)_: List of
            operators for which the expectation value is computed. Defaults to `None`.
        tmeas _(1d array-like, optional)_: Times between which measurement signals are
            averaged and saved. Defaults to `tsave`.
        ntrajs _(int, optional)_: Number of stochastic trajectories to solve
            concurrently. Defaults to 1.
        seed _(int, optional)_: Seed for the random number generator used to numerically
            sample the Wiener processes. Defaults to `None`, in which case the generator
            is seeded using a non-deterministic random number (see
            [`torch.Generator.seed()`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.seed)).
        solver _(Solver, optional)_: Solver for the SME integration (see the list
            below). Defaults to `None`.
        gradient _(Gradient, optional)_: Algorithm used to compute the gradient (see
            the list below). Defaults to `None`.
        options _(dict, optional)_: Generic options (see the list below). Defaults to
            `None`.

    Note-: Available solvers
      - `dq.solver.Euler`: Euler method (fixed step size SDE solver), not recommended
         except for testing purposes.
      - `dq.solver.Rouchon1`: Rouchon method of order 1 (fixed step size SDE solver).

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
            complex-valued tensors are converted. `tsave` and `tmeas` are converted to a
            real data type of the corresponding precision. Defaults to the complex data
            type set by `torch.set_default_dtype`.
        - **device** _(torch.device, optional)_ – Device on which the tensors are
            stored. Defaults to the device set by `torch.set_default_device`.

    Returns:
        Object holding the results of the stochastic master equation integration. It has
            the following attributes:

            - **states** _(Tensor)_ – Saved states with shape
                _(bH?, brho?, ntrajs, len(tsave), n, n)_.
            - **measurements** _(Tensor)_ - Saved measured signals with shape
                _(bH?, brho?, ntrajs, M, len(tmeas) - 1)_ where _M_ is the number of
                jump operators with non-zero efficiency.
            - **expects** _(Tensor, optional)_ – Saved expectation values with shape
                _(bH?, brho?, ntrajs, len(exp_ops), len(tsave))_.
            - **tsave** or **times** _(Tensor)_ – Times for which states and expectation
                values were saved.
            - **tmeas** _(Tensor)_ – Time intervals for which measured signals were
                averaged.
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
    """  # noqa: E501
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (ysave, exp_save, meas_save) with
    #    - ysave: (b_H?, b_rho0?, ntrajs, len(tsave), n, n)
    #    - exp_save: (b_H?, b_rho0?, ntrajs, len(exp_ops), len(tsave))
    #    - meas_save: (b_H?, b_rho0?, ntrajs, len(meas_ops), len(tmeas) - 1)

    # default solver
    if solver is None:
        raise ValueError(
            'No default solver yet, please specify one using the `solver` argument.'
        )

    # options
    options = Options(solver=solver, gradient=gradient, options=options)

    # solver class
    solvers = {
        Euler: SMEEuler,
        Rouchon1: SMERouchon1,
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
    solver = SOLVER_CLASS(
        H,
        rho0,
        tsave,
        tmeas,
        exp_ops,
        options,
        jump_ops=jump_ops,
        etas=etas,
        generator=generator,
    )

    # compute the result
    result = solver.run()

    # get saved tensors and restore correct batching
    if result.ysave is not None:
        result.ysave = result.ysave.squeeze(0, 1)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(0, 1)
    if result.meas_save is not None:
        result.meas_save = result.meas_save.squeeze(0, 1)

    return result
