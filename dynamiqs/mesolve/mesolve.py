from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from .._checks import check_shape, check_times
from .._utils import cdtype
from ..core._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_solver_class,
)
from ..gradient import Gradient
from ..options import Options
from ..result import MEResult
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import Shape, TimeArray
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEDopri8, MEEuler, METsit5
from .mepropagator import MEPropagator

__all__ = ['mesolve']


def mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    rho0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MEResult:
    r"""Solve the Lindblad master equation.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the Lindblad master
    equation ($\hbar=1$)
    $$
        \frac{\dd\rho(t)}{\dt} = -i[H(t), \rho(t)]
        + \sum_{k=1}^N \left(
            L_k(t) \rho(t) L_k^\dag(t)
            - \frac{1}{2} L_k^\dag(t) L_k(t) \rho(t)
            - \frac{1}{2} \rho(t) L_k^\dag(t) L_k(t)
        \right),
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators at time $t$.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`][dynamiqs.constant],
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See
        the [Time-dependent operators](../../tutorials/time-dependent-operators.md)
        tutorial for more details.

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the initial density
        matrix `rho0` can be batched to solve multiple master equations concurrently.
        All other arguments are common to every batch. See the
        [Batching simulations](../../tutorials/batching-simulations.md) tutorial for
        more details.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, each of shape (...Lk, n, n))_:
            List of jump operators.
        rho0 _(array-like of shape (...rho0, n, 1) or (...rho0, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] (supported:
            [`Tsit5`][dynamiqs.solver.Tsit5], [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Euler`][dynamiqs.solver.Euler],
            [`Rouchon1`][dynamiqs.solver.Rouchon1],
            [`Rouchon2`][dynamiqs.solver.Rouchon2],
            [`Propagator`][dynamiqs.solver.Propagator]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.MEResult`][dynamiqs.MEResult] object holding the result of the Lindblad
            master  equation integration. Use the attributes `states` and `expects`
            to access saved quantities, more details in
            [`dq.MEResult`][dynamiqs.MEResult].
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    rho0 = jnp.asarray(rho0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_mesolve_args(H, jump_ops, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # === convert rho0 to density matrix
    rho0 = todm(rho0)

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_mesolve(
        H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options
    )


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_mesolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    rho0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEResult:
    # === vectorize function
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.

    # the result is vectorized over `_saved` and `infos` and final_time
    out_axes = MEResult(None, None, None, None, 0, 0, 0)

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(
            H.shape[:-2], rho0.shape[:-2], *[jump_op.shape[:-2] for jump_op in jump_ops]
        )

        def broadcast(x: TimeArray) -> TimeArray:
            return x.broadcast_to(*(broadcast_shape + x.shape[-2:]))

        H = broadcast(H)
        jump_ops = list(map(broadcast, jump_ops))
        rho0 = jnp.broadcast_to(rho0, broadcast_shape + rho0.shape[-2:])

    n_batch = (
        H.in_axes,
        [jump_op.in_axes for jump_op in jump_ops],
        Shape(rho0.shape[:-2]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_mesolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_mesolve, n_batch, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options)


def _mesolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    rho0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEResult:
    # === select solver class
    solvers = {
        Euler: MEEuler,
        Dopri5: MEDopri5,
        Dopri8: MEDopri8,
        Tsit5: METsit5,
        Propagator: MEPropagator,
    }
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(tsave, rho0, H, exp_ops, solver, gradient, options, jump_ops)

    # === run solver
    result = solver.run()

    # === return result
    return result  # noqa: RET504


def _check_mesolve_args(
    H: TimeArray, jump_ops: list[TimeArray], rho0: Array, exp_ops: Array | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check rho0 shape
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'len(exp_ops)'})
