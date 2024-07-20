from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core.expm_solver import ExpmSolver
from ..gradient import Gradient
from ..options import Options
from ..result import PropagatorResult, Saved
from ..sesolve import sesolve
from ..solver import Solver, Expm, Tsit5
from ..time_array import TimeArray, ConstantTimeArray, PWCTimeArray
from ..utils.operators import eye

__all__ = ["propagator"]


def propagator(
    H: ArrayLike | TimeArray,
    tsave: ArrayLike,
    *,
    solver: Solver = None,  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> PropagatorResult:
    r"""Solve the Schrödinger equation by computing the propagator.

    This computation is done in one of two ways. If `solver` is set to `None`
    and if the Hamiltonian is of type [`dq.ConstantTimeArray`][dynamiqs.ConstantTimeArray] or
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
        solver: Solver for the integration. As mentioned above, if this is `None`
            and the Hamiltonian is constant or piece-wise constant, the propagator is computed
            by matrix exponential. If it is `None` and the Hamiltonian is not
            constant or piece-wise constant, the propagator is computed using
            [`dq.sesolve`][dynamiqs.sesolve] with the solver defaulting to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5]. If the solver is specified, the propagator
            is computed using [`dq.sesolve`][dynamiqs.sesolve]. See [`dq.sesolve`][dynamiqs.sesolve]
            for a list of supported solvers.

        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.PropagatorResult`][dynamiqs.PropagatorResult] object holding the result of the
            propagator computation. Use the attribute `propagator`
            to access saved quantities, more details in
            [`dq.PropagatorResult`][dynamiqs.PropagatorResult].
    """  # noqa: E501
    if not options.cartesian_batching:
        raise ValueError(
            "Flat batching not supported for propagator. "
            "Only cartesian batching is supported"
        )
    if solver is None and (
        isinstance(H, ConstantTimeArray) or isinstance(H, PWCTimeArray)
    ):
        solver = Expm()
        solver.assert_supports_gradient(gradient)
        solver_class = ExpmSolver(tsave, None, H, None, solver, gradient, options)
        result = solver_class.run()
        return result
    else:
        solver = Tsit5() if solver is None else solver
        initial_states = eye(H.shape[0])[..., None]
        seresult = sesolve(
            H, initial_states, tsave, solver=solver, gradient=gradient, options=options
        )
        if options.save_states:
            # indices are ...i, t, j. Want to permute them to
            # t, j, i such that the t index is first and each
            # column of the propogator corresponds to each initial state
            ndim = len(seresult.states.shape) - 1
            perm = list(range(ndim - 3)) + [ndim - 2, ndim - 1, ndim - 3]
            propagators = jnp.transpose(seresult.states[..., 0], perm)
        else:
            # otherwise, sesolve has only saved the final states
            # so we only need to permute the final two axes
            propagators = seresult.states[..., 0].swapaxes(-1, -2)
        saved = Saved(propagators, None, None)
        return PropagatorResult(tsave, solver, gradient, options, saved, seresult.infos)
