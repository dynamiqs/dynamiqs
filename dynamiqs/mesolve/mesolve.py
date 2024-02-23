from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import TimeArray
from ..utils.array_types import cdtype
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEDopri8, MEEuler, METsit5
from .mepropagator import MEPropagator


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
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
) -> Result:
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
        to a time-array using [`dq.totime`](/python_api/totime/totime.html).

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the initial density
        matrix `rho0` can be batched to solve multiple master equations concurrently.
        All other arguments are common to every batch.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, bL?, n, n))_: List
            of jump operators.
        rho0 _(array-like of shape (brho?, n, 1) or (brho?, n, n))_: Initial state.
        tsave _(array-like of shape (nt,))_: Times at which the states and expectation
            values are saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or
            from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5()`](/python_api/solver/Tsit5.html).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`](/python_api/options/Options.html).

    Returns:
        [`dq.Result`](/python_api/result/Result.html) object holding the result of the
            Lindblad master equation integration. It has the following attributes:

            - **states** _(array of shape (bH?, brho?, nt, n, n))_ -- Saved states.
            - **expects** _(array of shape (bH?, brho?, nE, nt), optional)_ -- Saved
                expectation values.
            - **extra** _(PyTree, optional)_ -- Extra data saved with `save_extra()` if
                specified in `options`.
            - **tsave** _(array of shape (nt,))_ -- Times for which states and
                expectation values were saved.
            - **solver** _(Solver)_ -- Solver used.
            - **gradient** _(Gradient)_ -- Gradient used.
            - **options** _(Options)_ -- Options used.
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(jump_op) for jump_op in jump_ops]
    rho0 = jnp.asarray(rho0, dtype=cdtype())
    rho0 = todm(rho0)
    tsave = jnp.asarray(tsave)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === vectorize function
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
    is_batched = (
        H.ndim > 2,
        False,  # todo: this is a temporary fix
        rho0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
    )
    # the result is vectorized over `saved`
    out_axes = Result(None, None, None, None, 0)

    f = compute_vmap(_mesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, rho0, tsave, exp_ops, solver, gradient, options)


def _mesolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    rho0: ArrayLike,
    tsave: ArrayLike,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> Result:
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
