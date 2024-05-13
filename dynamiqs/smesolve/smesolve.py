# ruff: noqa: ARG001

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from .._checks import check_shape, check_times
from .._utils import cdtype
from ..core._utils import (
    _astimearray,
    compute_vmap,
    get_solver_class,
    is_timearray_batched,
)
from ..gradient import Gradient
from ..options import Options
from ..result import SMEResult
from ..solver import Euler, Milstein, Solver
from ..time_array import TimeArray
from ..utils.utils import todm
from .smediffrax import SMEEuler, SMEMilstein

__all__ = ['smesolve']


def smesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    etas: ArrayLike,
    rho0: ArrayLike,
    tsave: ArrayLike,
    keys: list[PRNGKeyArray],
    *,
    tmeas: ArrayLike | None = None,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Milstein(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SMEResult:
    r"""Solve the diffusive stochastic master equation (SME).

    Warning:
        This function is still under test and development, it is not as stable as
        [`dq.sesolve()`][dynamiqs.sesolve] and [`dq.mesolve()`][dynamiqs.mesolve].

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho(t=0)$, according to the diffusive SME in Itô
    form ($\hbar=1$)
    $$
        \begin{split}
            \dd\rho(t) =&~ -i[H(t), \rho(t)] \dt \\\\
            &+ \sum_{k=1}^N \left(
                L_k(t) \rho(t) L_k^\dag(t)
                - \frac{1}{2} L_k^\dag(t) L_k(t) \rho(t)
                - \frac{1}{2} \rho(t) L_k^\dag(t) L_k(t)
            \right)\dt \\\\
            &+ \sum_{k=1}^N \sqrt{\eta_k} \left(
                L_k(t) \rho(t)
                + \rho(t) L_k^\dag(t)
                - \tr{(L_k(t)+L_k^\dag(t))\rho(t)}\rho(t)\ \dd W_k(t)
            \right),
        \end{split}
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$, $\{L_k(t)\}$ is a collection
    of jump operators at time $t$, each continuously measured with efficiency
    $0\leq\eta_k\leq1$ ($\eta_k=0$ for purely dissipative loss channels) and
    $\dd W_k(t)$ are independent Wiener processes.

    Notes:
        In quantum optics the _diffusive_ SME corresponds to homodyne or heterodyne
        detection schemes, as opposed to the _jump_ SME which corresponds to photon
        counting schemes. No solver for the jump SME is provided yet, if this is needed
        don't hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    The measured signals $I_k(t)=\dd Y_k(t)/\dt$ are defined by:
    $$
        \dd Y_k(t) = \sqrt{\eta_k} \tr{(L_k(t) + L_k^\dag(t)) \rho(t)} \dt + \dd W_k(t).
    $$

    Notes:
        Sometimes the signals are defined with a different but equivalent normalisation
        $\dd Y_k'(t) = \dd Y_k(t)/(2\sqrt{\eta_k})$.

    The signals $I_k(t)$ are singular quantities, the solver returns the averaged
    signals $J_k(t)$ defined for a time interval $[t_0, t_1[$ by:
    $$
        J_k([t_0, t_1[) = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} I_k(t)\ \dt
        = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} \dd Y_k(t).
    $$
    The time intervals for integration are defined by the argument `tmeas`, which
    defines `len(tmeas) - 1` intervals. By default, `tmeas = tsave`, so the signals
    are averaged between the times at which the states are saved.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`][dynamiqs.constant],
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See
        the [Time-dependent operators](../../tutorials/time-dependent-operators.md)
        tutorial for more details.

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H` and the initial density matrix `rho0` can be batched to
        solve multiple SMEs concurrently. All other arguments are common to every batch.
        See the [Batching simulations](../../tutorials/batching-simulations.md)
        tutorial for more details

    Warning:
        Batching on `jump_ops` and `etas is not yet supported, if this is needed don't
        hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    Warning:
        For now the same key is used for every batched simulation over `H` and `rho0`.
        This will be probably change in the future.

    Args:
        H _(array-like or time-array of shape (nH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (n, n))_: List of
            jump operators.
        etas _(array-like of shape (nL,))_: Measurement efficiencies, of the same
            length as `jump_ops` with values between 0 and 1. For a purely dissipative
            loss channel, set the corresponding efficiency to 0. No measurement signal
            will be returned for dissipative channels.
        rho0 _(array-like of shape (nrho0?, n, 1) or (nrho0?, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        keys _(list of PRNG key, of shape (ntrajs,))_: PRNG keys used as the random
            keys to sample the Wiener processes. The number of keys defines the number
            of sampled stochastic trajectories.
        tmeas _(array-like of shape (ntmeas,), optional)_: Times between which
            measurement signals are averaged and saved. Defaults to `tsave`.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Milstein`][dynamiqs.solver.Milstein] (supported:
            [`Milstein`][dynamiqs.solver.Milstein], [`Euler`][dynamiqs.solver.Euler]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SMEResult`][dynamiqs.SMEResult] object holding the result of the SME
            integration. Use the attributes `states`, `measurements` and `expects`
            to access saved quantities, more details in
            [`dq.SMEResult`][dynamiqs.SMEResult].
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    etas = jnp.asarray(etas)
    rho0 = jnp.asarray(rho0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    tmeas = jnp.asarray(tmeas) if tmeas is not None else tsave
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_smesolve_args(H, jump_ops, etas, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')
    tmeas = check_times(tmeas, 'tmeas')

    # === convert rho0 to density matrix
    rho0 = todm(rho0)

    # === split jump operators
    # split between purely dissipative (eta = 0) and monitored (eta != 0)
    dissipative_ops = [L for L, eta in zip(jump_ops, etas) if eta == 0]
    measured_ops = [L for L, eta in zip(jump_ops, etas) if eta != 0]
    etas = etas[etas != 0]

    # we implement the jitted vmap in another function to pre-convert QuTiP objects
    # (which are not JIT-compatible) to JAX arrays
    return _vmap_smesolve(
        H,
        dissipative_ops,
        measured_ops,
        etas,
        rho0,
        tsave,
        keys,
        tmeas,
        exp_ops,
        solver,
        gradient,
        options,
    )


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vmap_smesolve(
    H: TimeArray,
    dissipative_ops: list[TimeArray],
    measured_ops: list[TimeArray],
    etas: Array,
    rho0: Array,
    tsave: Array,
    keys: PRNGKeyArray,
    tmeas: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SMEResult:
    # the result is vectorized over `_saved`, `infos` and `keys`
    out_axes = SMEResult(None, None, None, None, 0, 0, None, 0)

    # === vectorize function over keys
    in_axes = (None, None, None, None, None, None, 0, None, None, None, None, None)
    f = jax.vmap(_smesolve_single_trajectory, in_axes=in_axes, out_axes=out_axes)

    # === vectorize function over H and rho0, all other arguments are not vectorized
    is_batched = (
        is_timearray_batched(H),
        False,
        False,
        False,
        rho0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    # compute vectorized function with given batching strategy
    f = compute_vmap(f, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(
        H,
        dissipative_ops,
        measured_ops,
        etas,
        rho0,
        tsave,
        keys,
        tmeas,
        exp_ops,
        solver,
        gradient,
        options,
    )


def _smesolve_single_trajectory(
    H: TimeArray,
    dissipative_ops: list[TimeArray],
    measured_ops: list[TimeArray],
    etas: Array,
    rho0: Array,
    tsave: Array,
    key: PRNGKeyArray,
    tmeas: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SMEResult:
    # === select solver class
    solvers = {Euler: SMEEuler, Milstein: SMEMilstein}
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(
        tsave,
        rho0,
        H,
        exp_ops,
        solver,
        gradient,
        options,
        tmeas,
        key,
        dissipative_ops,
        measured_ops,
        etas,
    )

    # === run solver
    result = solver.run()

    # === return result
    return result  # noqa: RET504


def _check_smesolve_args(
    H: TimeArray,
    jump_ops: list[TimeArray],
    etas: Array,
    rho0: Array,
    exp_ops: Array | None,
):
    # === check H shape
    check_shape(H, 'H', '(?, n, n)', subs={'?': 'nH?'})

    # === check jump_ops
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(n, n)')

    if len(jump_ops) == 0:
        logging.warn(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check etas
    check_shape(etas, 'etas', '(n,)', subs={'n': 'nL'})

    if not len(etas) == len(jump_ops):
        raise ValueError(
            'Argument `etas` should be of the same length as argument `jump_ops`, but'
            f' len(etas)={len(etas)} and len(jump_ops)={len(jump_ops)}.'
        )

    if jnp.all(etas == 0):
        raise ValueError(
            'Argument `etas` contains only null values, consider using `dq.mesolve()`'
            ' to solve the Lindblad master equation.'
        )

    if not (jnp.all(etas >= 0) and jnp.all(etas <= 1)):
        raise ValueError(
            'Argument `etas` should only contain values between 0 and 1, but'
            f' is {etas}.'
        )

    # === check rho0 shape
    check_shape(rho0, 'rho0', '(?, n, 1)', '(?, n, n)', subs={'?': 'nrho0?'})

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'nE'})
