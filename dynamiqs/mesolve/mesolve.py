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
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    r"""Solve the Lindblad master equation.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the Lindblad master
    equation ($\hbar=1$):
    $$
        \frac{\dd\rho}{\dt} =-i[H(t), \rho]
        + \sum_{k=1}^N \left(
            L_k(t) \rho L_k^\dag(t)
            - \frac{1}{2} L_k^\dag(t) L_k(t) \rho
            - \frac{1}{2} \rho L_k^\dag(t) L_k(t)
        \right),
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or jump operators depend on time, they can be converted to a
        `TimeArray` using [`dq.totime`](/python_api/totime/totime.html).

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the initial density matrix `rho0` and the jump operators
        `L_k` can be batched to solve multiple master equations concurrently. All other
        arguments are common to every batch.

    Args:
        H _(array-like or TimeArray)_: Hamiltonian of shape _(bH?, n, n)_.
        jump_ops _(list of array-like or TimeArray)_: List of jump operators, each of
            shape _(bL?, n, n)_.
        psi0 _(array-like)_: Initial state vector of shape _(brho?, n, 1)_ or density
            matrix of shape _(brho?, n, n)_.
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
            the Lindblad master equation integration. It has the following attributes:

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
    # we vectorize over H, jump_ops and psi0, all other arguments are not vectorized
    jump_ops_ndim = _astimearray(jump_ops[0]).ndim + 1
    is_batched = (
        H.ndim > 2,
        jump_ops_ndim > 3,  # todo: this is a temporary fix
        psi0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
    )
    # the result is vectorized over ysave and Esave
    out_axes = Result(None, None, None, None, 0, 0)

    f = compute_vmap(_mesolve, options.cartesian_batching, is_batched, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, psi0, tsave, exp_ops, solver, gradient, options)


def _mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),
    gradient: Gradient | None = None,
    options: Options = Options(),
) -> Result:
    # === convert arguments
    H = _astimearray(H)
    Ls = [_astimearray(L) for L in jump_ops]
    y0 = jnp.asarray(psi0, dtype=cdtype())
    y0 = todm(y0)
    ts = jnp.asarray(tsave)
    Es = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

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
    solver = solver_class(ts, y0, H, Es, solver, gradient, options, Ls)

    # === run solver
    result = solver.run()

    # === return result
    return result
