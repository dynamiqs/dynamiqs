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
from ...time_array import TimeArray
from .._utils import _astimearray, catch_xla_runtime_error, get_integrator_class, ispwc
from ..sepropagator.dynamiqs_integrator import SEPropagatorDynamiqsIntegrator
from ..sepropagator.expm_integrator import SEPropagatorExpmIntegrator


def sepropagator(
    H: ArrayLike | TimeArray,
    tsave: ArrayLike,
    *,
    solver: Solver = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEPropagatorResult:
    r"""Compute the propagator of the Schrödinger equation.

    This computation is done in one of two ways. If `solver` is set to `None`
    and if the Hamiltonian is of type
    [`dq.ConstantTimeArray`][dynamiqs.ConstantTimeArray] or
    [`dq.PWCTimeArray`][dynamiqs.PWCTimeArray], then the propagator is computed
    by appropriately exponentiating the Hamiltonian. On the other hand if the solver is
    specified or if the Hamiltonian is not constant or piece-wise constant, we solve
    the Schrödinger equation using [`dq.sesolve`][dynamiqs.sesolve] and batching over
    all initial states.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators
            are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        solver: Solver for the integration. If solver is `None` (default value) then we
            redirect to [`dq.solver.Expm`][dynamiqs.solver.Expm] or
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] depending on the type of
            the input Hamiltonian. See [`dq.sesolve`][dynamiqs.sesolve]
            for a list of supported solvers.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SEPropagatorResult`][dynamiqs.SEPropagatorResult] object holding
            the result of the propagator computation. Use the attribute `propagator`
            to access saved quantities, more details in
            [`dq.SEPropagatorResult`][dynamiqs.SEPropagatorResult].
    """
    # === convert arguments
    H = _astimearray(H)
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_sepropagator_args(H)
    tsave = check_times(tsave, 'tsave')

    return _sepropagator(H, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _sepropagator(
    H: TimeArray,
    tsave: Array,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    # === select and check integrator class
    if solver is None:  # default solver
        solver = Expm() if ispwc(H) else Tsit5()
    integrators = {
        Expm: SEPropagatorExpmIntegrator,
        Euler: SEPropagatorDynamiqsIntegrator,
        Dopri5: SEPropagatorDynamiqsIntegrator,
        Dopri8: SEPropagatorDynamiqsIntegrator,
        Tsit5: SEPropagatorDynamiqsIntegrator,
        Kvaerno3: SEPropagatorDynamiqsIntegrator,
        Kvaerno5: SEPropagatorDynamiqsIntegrator,
    }
    integrator_class = get_integrator_class(integrators, solver)
    solver.assert_supports_gradient(gradient)
    integrator = integrator_class(tsave, None, H, None, solver, gradient, options)

    # === run integrator and return result
    return integrator.run()


def _check_sepropagator_args(H: TimeArray):
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})
