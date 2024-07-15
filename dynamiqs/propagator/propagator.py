from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..gradient import Gradient
from ..options import Options
from ..result import PropagatorResult
from ..sesolve import sesolve
from ..solver import Solver, Tsit5
from ..time_array import TimeArray
from ..utils.operators import eye

__all__ = ['propagator']


def propagator(
    H: ArrayLike | TimeArray,
    tsave: ArrayLike,
    *,
    jump_ops: list[ArrayLike | TimeArray] = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> PropagatorResult:
    r"""Solve the Schrödinger equation by computing the propagator.
     
    Warning:
        As of now, this function only supports unitary evolution and thus 
        does not yet support collapse operators
        
    This function computes the propagator by batching over all initial states, and
    then appropriately combining the results. For more information on how to 
    solve for the time evolution of each individual state, see [`dq.sesolve`][dynamiqs.sesolve]

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators 
            are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        jump_ops _(list of array-like or time-array, each of shape (...Lk, n, n))_:
            List of jump operators.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] (supported:
            [`Tsit5`][dynamiqs.solver.Tsit5], [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Kvaerno3`][dynamiqs.solver.Kvaerno3],
            [`Kvaerno5`][dynamiqs.solver.Kvaerno5],
            [`Euler`][dynamiqs.solver.Euler],
            [`Propagator`][dynamiqs.solver.Propagator]).

        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.PropagatorResult`][dynamiqs.PropagatorResult] object holding the result of the
            propagator computation. Use the attribute `propagators`
            to access saved quantities, more details in
            [`dq.PropagatorResult`][dynamiqs.PropagatorResult].
    """  # noqa: E501
    if jump_ops is None:
        jump_ops = []
    if len(jump_ops) != 0:
        raise NotImplementedError("propagator only implemented for the unitary case")
    if not options.cartesian_batching:
        raise ValueError("flat batching not supported for propagator."
                         " Only cartesian batching is supported")
    dim = H.shape[0]
    initial_states = eye(dim)[..., None]

    seresult = sesolve(
        H,
        initial_states,
        tsave,
        exp_ops=None,
        solver=solver,
        gradient=gradient,
        options=options
    )
    if options.save_states:
        # indices are ...i, t, j. Want to permute them to
        # t, j, i such that the t index is first and each
        # column of the propogator corresponds to each initial state
        ndim = len(seresult.states.shape) - 1
        perm = list(range(ndim - 3)) + [ndim - 2, ndim - 1, ndim - 3]
        propagators = jnp.transpose(seresult.states[..., 0], perm)
    else:
        propagators = seresult.states[..., 0].swapaxes(-1, -2)
    return PropagatorResult(
        tsave, solver, gradient, options, seresult._saved, seresult.infos, propagators
    )
