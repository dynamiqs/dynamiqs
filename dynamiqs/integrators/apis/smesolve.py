# ruff: noqa: ARG001

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from ..._checks import check_shape, check_times
from ..._utils import cdtype
from ...gradient import Gradient
from ...options import Options
from ...result import SMESolveResult
from ...solver import Euler, Milstein, Solver
from ...time_array import Shape, TimeArray
from ...utils.quantum_utils.general import todm
from .._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_integrator_class,
)
from ..smesolve.diffrax_integrator import (
    SMESolveEulerIntegrator,
    SMESolveMilsteinIntegrator,
)


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
) -> SMESolveResult:
    r"""Solve the diffusive stochastic master equation (SME).

    Warning:
        This function is still under test and development. Please
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new) if
        you encounter any bug.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the diffusive SME in Itô
    form (with $\hbar=1$ and where time is implicit(1))
    $$
        \begin{split}
            \dd\rho =&~ -i[H, \rho]\,\dt + \sum_{k=1}^N \left(
                L_k \rho L_k^\dag
                - \frac{1}{2} L_k^\dag L_k \rho
                - \frac{1}{2} \rho L_k^\dag L_k
        \right)\dt \\\\
            &+ \sum_{k=1}^N \sqrt{\eta_k} \left(
                L_k \rho
                + \rho L_k^\dag
                - \tr{(L_k+L_k^\dag)\rho}\rho
            \right)\dd W_k,
        \end{split}
    $$
    where $H$ is the system's Hamiltonian, $\{L_k\}$ is a collection of jump operators,
    each continuously measured with efficiency $0\leq\eta_k\leq1$ ($\eta_k=0$ for
    purely dissipative loss channels) and $\dd W_k$ are independent Wiener processes.
    { .annotate }

    1. With explicit time dependence:
        - $\rho\to\rho(t)$
        - $H\to H(t)$
        - $L_k\to L_k(t)$
        - $\dd W_k\to \dd W_k(t)$

    Note-: Diffusive vs. jump SME
        In quantum optics the _diffusive_ SME corresponds to homodyne or heterodyne
        detection schemes, as opposed to the _jump_ SME which corresponds to photon
        counting schemes. No solver for the jump SME is provided yet, if this is needed
        don't hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    The measured signals $I_k=\dd Y_k/\dt$ are defined by (again time is implicit):
    $$
        \dd Y_k = \sqrt{\eta_k} \tr{(L_k + L_k^\dag) \rho} \dt + \dd W_k.
    $$

    Note-: Signal normalisation
        Sometimes the signals are defined with a different but equivalent normalisation
        $\dd Y_k' = \dd Y_k/(2\sqrt{\eta_k})$.

    The signals $I_k$ are singular quantities, the solver returns the averaged signals
    $J_k$ defined for a time interval $[t_0, t_1)$ by:
    $$
        J_k([t_0, t_1)) = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} I_k(t)\, \dt
        = \frac{1}{t_1-t_0}\int_{t_0}^{t_1} \dd Y_k(t).
    $$
    The time intervals for integration are defined by the argument `tmeas`, which
    defines `len(tmeas) - 1` intervals. By default, `tmeas = tsave`, so the signals
    are averaged between the times at which the states are saved.

    Note-: Defining a time-dependent Hamiltonian or jump operator
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
        [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H` and the initial density matrix `rho0` can be batched to
        solve multiple SMEs concurrently. All other arguments are common to every batch.
        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.

    Warning:
            Batching on `jump_ops` and `etas` is not yet supported, if this is needed
            don't hesitate to
            [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, each of shape (n, n))_: List of
            jump operators.
        etas _(array-like of shape (len(jump_ops),))_: Measurement efficiency for each
            loss channel with values between 0 (purely dissipative) and 1 (perfectly
            monitored). No measurement signal is returned for purely dissipative
            loss channels.
        rho0 _(array-like of shape (...rho0, n, 1) or (...rho0, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        keys _(list of PRNG keys)_: PRNG keys used to sample the Wiener processes.
            The number of elements defines the number of sampled stochastic
            trajectories.
        tmeas _(array-like of shape (ntmeas,), optional)_: Times between which
            measurement signals are averaged and saved. Defaults to `tsave`.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Milstein`][dynamiqs.solver.Milstein] (supported:
            [`Milstein`][dynamiqs.solver.Milstein], [`Euler`][dynamiqs.solver.Euler]).
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SMESolveResult`][dynamiqs.SMESolveResult] object holding the result of the
            SME integration. Use the attributes `states`, `measurements` and `expects`
            to access saved quantities, more details in
            [`dq.SMESolveResult`][dynamiqs.SMESolveResult].
    """  # noqa: E501
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    etas = jnp.asarray(etas)
    rho0 = jnp.asarray(rho0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    tmeas = jnp.asarray(tmeas) if tmeas is not None else tsave
    if exp_ops is not None:
        exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if len(exp_ops) > 0 else None

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

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_smesolve(
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


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_smesolve(
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
) -> SMESolveResult:
    # === vectorize function
    # we vectorize over H and rho0, all other arguments are not vectorized
    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.

    # the result is vectorized over `_saved` and `infos`
    out_axes = SMESolveResult(False, False, False, False, 0, 0, False, 0)

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(H.shape[:-2], rho0.shape[:-2])

        def broadcast(x: TimeArray) -> TimeArray:
            return x.broadcast_to(*(broadcast_shape + x.shape[-2:]))

        H = broadcast(H)
        rho0 = jnp.broadcast_to(rho0, broadcast_shape + rho0.shape[-2:])

    n_batch = (
        H.in_axes,
        Shape(),
        Shape(),
        Shape(),
        Shape(rho0.shape[:-2]),
        Shape(),
        Shape([len(keys)]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_smesolve_single_trajectory, n_batch, out_axes)
    else:
        f = _flat_vectorize(_smesolve_single_trajectory, n_batch, out_axes)

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
) -> SMESolveResult:
    # === select integrator class
    integrators = {Euler: SMESolveEulerIntegrator, Milstein: SMESolveMilsteinIntegrator}
    integrator_class = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    integrator = integrator_class(
        tsave,
        rho0,
        H,
        solver,
        gradient,
        options,
        tmeas,
        key,
        dissipative_ops,
        measured_ops,
        etas,
        exp_ops,
    )

    # === run solver
    result = integrator.run()

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
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(n, n)')

    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check etas
    check_shape(etas, 'etas', '(n,)', subs={'n': 'len(jump_ops)'})

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
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'len(exp_ops)'})
