from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options
from ...result import MEPropagatorResult
from ...solver import Expm, Solver
from ...time_array import Shape, TimeArray
from ...utils.operators import eye
from .._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_integrator_class,
)
from ..mepropagator.expm_integrator import MEPropagatorExpmIntegrator


def mepropagator(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    tsave: ArrayLike,
    *,
    solver: Solver = Expm(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MEPropagatorResult:
    r"""Compute the propagator of the Lindblad master equation.

    This function computes the propagator $\mathcal{U}(t)$ at time $t$ of the Lindblad
    master equation (with $\hbar=1$)
    $$
        \mathcal{U}(t) = \mathscr{T}\exp\left(\int_0^t\mathcal{L}(t')\dt'\right),
    $$
    where $\mathscr{T}$ is the time-ordering symbol and $\mathcal{L}$ is the system's
    Liouvillian. The formula simplifies to $\mathcal{U}(t)=e^{t\mathcal{L}}$ if the
    Liouvillian does not depend on time.

    Warning:
        This function only supports constant or piecewise constant Hamiltonians and jump
        operators. Support for arbitrary time dependence will be added soon.

    Note-: Defining a time-dependent Hamiltonian or jump operator
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
        [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H` and the jump operators `jump_ops` can be batched to compute
        multiple propagators concurrently. All other arguments are common to every
        batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, each of shape (...Lk, n, n))_:
            List of jump operators.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators are
            saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or from `t0`
            to `tsave[-1]` if `t0` is specified in `options`.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Expm`][dynamiqs.solver.Expm] (explicit matrix exponentiation),
            which is the only supported solver for now.
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.MEPropagatorResult`][dynamiqs.MEPropagatorResult] object holding
            the result of the propagator computation. Use the attribute
            `propagators` to access saved quantities, more details in
            [`dq.MEPropagatorResult`][dynamiqs.MEPropagatorResult].
    """  # noqa: E501
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_mepropagator_args(H, jump_ops)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_mepropagator(H, jump_ops, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_mepropagator(
    H: TimeArray,
    jump_ops: list[TimeArray],
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # === vectorize function
    # we vectorize over H and jump_ops, all other arguments are not vectorized
    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.

    # the result is vectorized over `_saved` and `infos`
    out_axes = MEPropagatorResult(False, False, False, False, 0, 0)

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(
            H.shape[:-2], *[jump_op.shape[:-2] for jump_op in jump_ops]
        )

        def broadcast(x: TimeArray) -> TimeArray:
            return x.broadcast_to(*(broadcast_shape + x.shape[-2:]))

        H = broadcast(H)
        jump_ops = list(map(broadcast, jump_ops))

    n_batch = (
        H.in_axes,
        [jump_op.in_axes for jump_op in jump_ops],
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_mepropagator, n_batch, out_axes)
    else:
        f = _flat_vectorize(_mepropagator, n_batch, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, tsave, solver, gradient, options)


def _mepropagator(
    H: TimeArray,
    jump_ops: list[TimeArray],
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # === select integrator class
    integrators = {Expm: MEPropagatorExpmIntegrator}
    integrator_class = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    y0 = eye(H.shape[-1] ** 2)
    integrator = integrator_class(tsave, y0, H, solver, gradient, options, jump_ops)

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mepropagator_args(H: TimeArray, jump_ops: list[TimeArray]):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sepropagator()`'
            ' to compute propagators for the Schrödinger equation.'
        )
