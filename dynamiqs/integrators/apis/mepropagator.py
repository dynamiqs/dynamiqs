from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options
from ...qarrays.layout import dense
from ...qarrays.qarray import QArrayLike
from ...result import MEPropagatorResult
from ...solver import Expm, Solver
from ...time_array import TimeArray
from ...utils.operators import eye
from .._utils import (
    _astimearray,
    cartesian_vmap,
    catch_xla_runtime_error,
    get_integrator_class,
    multi_vmap,
)
from ..core.abstract_integrator import MEPropagatorIntegrator
from ..mepropagator.expm_integrator import MEPropagatorExpmIntegrator


def mepropagator(
    H: QArrayLike | TimeArray,
    jump_ops: list[QArrayLike | TimeArray],
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
        H _(qarray-like or time-array of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-array, each of shape (...Lk, n, n))_:
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
    Ls = [_astimearray(L) for L in jump_ops]
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_mepropagator_args(H, Ls)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_mepropagator(H, Ls, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_mepropagator(
    H: TimeArray,
    Ls: list[TimeArray],
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # vectorize input over H and Ls
    in_axes = (H.in_axes, [L.in_axes for L in Ls], None, None, None, None)
    # vectorize output over `_saved` and `infos`
    out_axes = MEPropagatorResult(None, None, None, None, 0, 0)

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], 0, 0, 0, 0)
        f = cartesian_vmap(_mepropagator, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        # vectorize the function
        f = multi_vmap(_mepropagator, in_axes, out_axes, nvmap)

    return f(H, Ls, tsave, solver, gradient, options)


def _mepropagator(
    H: TimeArray,
    Ls: list[TimeArray],
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # === select integrator class
    integrators = {Expm: MEPropagatorExpmIntegrator}
    integrator_class: MEPropagatorIntegrator = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    y0 = eye(H.shape[-1] ** 2, layout=dense)
    integrator = integrator_class(
        ts=tsave, y0=y0, solver=solver, gradient=gradient, options=options, H=H, Ls=Ls
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mepropagator_args(H: TimeArray, Ls: list[TimeArray]):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sepropagator()`'
            ' to compute propagators for the Schrödinger equation.'
        )
