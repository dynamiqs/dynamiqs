from __future__ import annotations

from jaxtyping import ArrayLike

from ...gradient import Gradient
from ...options import Options
from ...result import PropagatorResult
from ...solver import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import TimeArray
from .._utils import get_integrator_class, ispwc
from ..sepropagator.diffrax_integrator import SEPropagatorDiffraxIntegrator
from ..sepropagator.expm_integrator import SEPropagatorExpmIntegrator


def sepropagator(
    H: ArrayLike | TimeArray,
    tsave: ArrayLike,
    *,
    solver: Solver = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> PropagatorResult:
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
        [`dq.PropagatorResult`][dynamiqs.PropagatorResult] object holding
            the result of the propagator computation. Use the attribute `propagator`
            to access saved quantities, more details in
            [`dq.PropagatorResult`][dynamiqs.PropagatorResult].
    """
    if solver is None:
        solver = Expm() if ispwc(H) else Tsit5()
    integrators = {
        Expm: SEPropagatorExpmIntegrator,
        Euler: SEPropagatorDiffraxIntegrator,
        Dopri5: SEPropagatorDiffraxIntegrator,
        Dopri8: SEPropagatorDiffraxIntegrator,
        Tsit5: SEPropagatorDiffraxIntegrator,
        Kvaerno3: SEPropagatorDiffraxIntegrator,
        Kvaerno5: SEPropagatorDiffraxIntegrator,
    }
    integrator_class = get_integrator_class(integrators, solver)
    solver.assert_supports_gradient(gradient)
    integrator = integrator_class(tsave, None, H, None, solver, gradient, options)
    return integrator.run()
