from __future__ import annotations

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
from ..result import SEResult
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import Shape, TimeArray
from .sediffrax import SEDopri5, SEDopri8, SEEuler, SETsit5
from .sepropagator import SEPropagator

__all__ = ['sesolve']


def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEResult:
    r"""Solve the Schrödinger equation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi_0}$, according to the Schrödinger
    equation ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt} = -i H(t) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be converted to a time-array using
        [`dq.constant()`][dynamiqs.constant], [`dq.pwc()`][dynamiqs.pwc],
        [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See
        the [Time-dependent operators](../../tutorials/time-dependent-operators.md)
        tutorial for more details.

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple Schrödinger equations concurrently. All other arguments are
        common to every batch. See the
        [Batching simulations](../../tutorials/batching-simulations.md) tutorial for
        more details.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        psi0 _(array-like of shape (...psi0, n, 1))_: Initial state.
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
            [`Propagator`][dynamiqs.solver.Propagator]).

        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SEResult`][dynamiqs.SEResult] object holding the result of the
            Schrödinger equation integration. Use the attributes `states` and `expects`
            to access saved quantities, more details in
            [`dq.SEResult`][dynamiqs.SEResult].
    """
    # === convert arguments
    H = _astimearray(H)
    psi0 = jnp.asarray(psi0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_sesolve_args(H, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_sesolve(H, psi0, tsave, exp_ops, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_sesolve(
    H: TimeArray,
    psi0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEResult:
    # === vectorize function
    # we vectorize over H and psi0, all other arguments are not vectorized

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(H.shape[:-2], psi0.shape[:-2])
        H = H.broadcast_to(*(broadcast_shape + H.shape[-2:]))
        psi0 = jnp.broadcast_to(psi0, broadcast_shape + psi0.shape[-2:])

    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.
    n_batch = (
        H.in_axes,
        Shape(psi0.shape[:-2]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # the result is vectorized over `_saved` and `infos`
    out_axes = SEResult(None, None, None, None, 0, 0)

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_sesolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_sesolve, n_batch, out_axes)

    # === apply vectorized function
    return f(H, psi0, tsave, exp_ops, solver, gradient, options)


def _sesolve(
    H: TimeArray,
    psi0: Array,
    tsave: Array,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEResult:
    # === select solver class
    solvers = {
        Euler: SEEuler,
        Dopri5: SEDopri5,
        Dopri8: SEDopri8,
        Tsit5: SETsit5,
        Propagator: SEPropagator,
    }
    solver_class = get_solver_class(solvers, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(tsave, psi0, H, exp_ops, solver, gradient, options)

    # === run solver
    result = solver.run()

    # === return result
    return result  # noqa: RET504


def _check_sesolve_args(H: TimeArray, psi0: Array, exp_ops: Array | None):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'len(exp_ops)'})
