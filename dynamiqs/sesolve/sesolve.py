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
from .sediffrax import SEDopri5, SEDopri8, SEEuler, SETsit5
from .sepropagator import SEPropagator


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    r"""Solve the Schrödinger equation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ starting
    from an initial state $\ket{\psi_0}$, according to the Schrödinger equation
    ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt} = -i H(t) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be converted to a `TimeArray` using
        [`dq.totime`](/python_api/totime/totime.html).

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple Schrödinger equations concurrently. All other arguments are
        common to every batch.

    Args:
        H _(array-like or TimeArray)_: Hamiltonian of shape _(bH?, n, n)_.
        psi0 _(array-like)_: Initial state vector of shape _(bpsi?, n, 1)_.
        tsave _(1D array-like)_: Times at which the states and expectation values are
            saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or from `t0`
            to `tsave[-1]` if `t0` is given in `options`.
        exp_ops _(list of 2D array-like, optional)_: List of operators of shape _(n, n)_
            for which the expectation value is computed. Defaults to `None`.
        solver _(Solver, optional)_: Solver for the differential equation integration.
            Defaults to `dq.solver.Tsit5()`.
        gradient _(Gradient, optional)_: Algorithm used to compute the gradient.
            Defaults to `None`.
        options _(Options, optional)_: Generic options. Defaults to `None`.

    Returns:
        Object of type [`Result`](/python_api/result/Result.html) holding the result of
            the Schrödinger equation integration. It has the following attributes:

            - **states** _(Array)_ – Saved states with shape
                _(bH?, bpsi?, len(tsave), n, 1)_.
            - **expects** _(Array, optional)_ – Saved expectation values with shape
                _(bH?, bpsi?, len(exp_ops), len(tsave))_.
            - **tsave** _(Array)_ – Times for which states and expectation values were
                saved.
            - **solver** (Solver) –  Solver used.
            - **gradient** (Gradient) – Gradient used.
            - **options** _(dict)_  – Options used.
    """
    # === vectorize function
    # we vectorize over H and psi0, all other arguments are not vectorized
    is_batched = (H.ndim > 2, psi0.ndim > 2, False, False, False, False, False)
    # the result is vectorized over ysave and Esave
    out_axes = Result(None, None, None, None, 0, 0)
    f = compute_vmap(_sesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(H, psi0, tsave, exp_ops, solver, gradient, options)


def _sesolve(
    H: ArrayLike | TimeArray,
    psi0: ArrayLike,
    tsave: ArrayLike,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    # === convert arguments
    H = _astimearray(H)
    y0 = jnp.asarray(psi0, dtype=cdtype())
    ts = jnp.asarray(tsave)
    Es = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

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
    solver = solver_class(ts, y0, H, Es, solver, gradient, options)

    # === run solver
    result = solver.run()

    # === return result
    return result
