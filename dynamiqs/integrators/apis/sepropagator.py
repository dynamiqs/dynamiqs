from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options
from ...result import SEPropagatorResult
from ...solver import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import Shape, TimeArray
from ...utils.operators import eye
from .._utils import (
    _astimearray,
    _cartesian_vectorize,
    catch_xla_runtime_error,
    get_integrator_class,
    ispwc,
)
from ..sepropagator.diffrax_integrator import (
    SEPropagatorDopri5Integrator,
    SEPropagatorDopri8Integrator,
    SEPropagatorEulerIntegrator,
    SEPropagatorKvaerno3Integrator,
    SEPropagatorKvaerno5Integrator,
    SEPropagatorTsit5Integrator,
)
from ..sepropagator.expm_integrator import SEPropagatorExpmIntegrator


def sepropagator(
    H: ArrayLike | TimeArray,
    tsave: ArrayLike,
    *,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEPropagatorResult:
    r"""Compute the propagator of the Schrödinger equation.

    This function computes the propagator $U(t)$ at time $t$ of the Schrödinger
    equation (with $\hbar=1$)
    $$
        U(t) = \mathscr{T}\exp\left(-i\int_0^tH(t')\dt'\right),
    $$
    where $\mathscr{T}$ is the time-ordering symbol and $H$ is the system's
    Hamiltonian. The formula simplifies to $U(t)=e^{-iHt}$ if the Hamiltonian
    does not depend on time.

    If the Hamiltonian is constant or piecewise constant, the propagator is
    computed by directly exponentiating the Hamiltonian. Otherwise, the
    propagator is computed by solving the Schrödinger equation with an ODE solver.

    Note-: Defining a time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be converted to a time-array using
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H` can be batched to compute multiple propagators
        concurrently. All other arguments are common to every batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators
            are saved. The equation is solved from `tsave[0]` to `tsave[-1]`,
            or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        solver: Solver for the integration. Defaults to `None` which redirects
            to [`dq.solver.Expm`][dynamiqs.solver.Expm] (explicit matrix
            exponentiation) or [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5]
            depending on the Hamiltonian type (supported:
            [`Expm`][dynamiqs.solver.Expm],
            [`Tsit5`][dynamiqs.solver.Tsit5],
            [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Kvaerno3`][dynamiqs.solver.Kvaerno3],
            [`Kvaerno5`][dynamiqs.solver.Kvaerno5],
            [`Euler`][dynamiqs.solver.Euler]).
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SEPropagatorResult`][dynamiqs.SEPropagatorResult] object holding
            the result of the propagator computation. Use the attribute
            `propagators` to access saved quantities, more details in
            [`dq.SEPropagatorResult`][dynamiqs.SEPropagatorResult].
    """  # noqa: E501
    # === convert arguments
    H = _astimearray(H)
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_sepropagator_args(H)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_sepropagator(H, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_sepropagator(
    H: TimeArray,
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    # === vectorize function
    # we vectorize over H, all other arguments are not vectorized
    n_batch = (H.in_axes, Shape(), Shape(), Shape(), Shape())

    # the result is vectorized over `_saved` and `infos`
    out_axes = SEPropagatorResult(False, False, False, False, 0, 0)

    # compute vectorized function
    f = _cartesian_vectorize(_sepropagator, n_batch, out_axes)

    # === apply vectorized function
    return f(H, tsave, solver, gradient, options)


def _sepropagator(
    H: TimeArray,
    tsave: Array,
    solver: Solver | None,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    # === select integrator class
    if solver is None:  # default solver
        solver = Expm() if ispwc(H) else Tsit5()

    integrators = {
        Expm: SEPropagatorExpmIntegrator,
        Euler: SEPropagatorEulerIntegrator,
        Dopri5: SEPropagatorDopri5Integrator,
        Dopri8: SEPropagatorDopri8Integrator,
        Tsit5: SEPropagatorTsit5Integrator,
        Kvaerno3: SEPropagatorKvaerno3Integrator,
        Kvaerno5: SEPropagatorKvaerno5Integrator,
    }
    integrator_class = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    y0 = eye(H.shape[-1])
    integrator = integrator_class(tsave, y0, H, solver, gradient, options)

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_sepropagator_args(H: TimeArray):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})
