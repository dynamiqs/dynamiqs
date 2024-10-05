from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp

# from jax.lax import while_loop
import optimistix as optx
from jax import Array
from jax.random import PRNGKey
from jaxtyping import ArrayLike
from optimistix import AbstractRootFinder

from ..._checks import check_shape, check_times
from ..._utils import cdtype
from ...gradient import Gradient
from ...options import Options
from ...result import MCSolveResult
from ...solver import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import Shape, TimeArray
from .._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    get_integrator_class,
)
from ..mcsolve.diffrax_integrator import (
    MCSolveDopri5Integrator,
    MCSolveDopri8Integrator,
    MCSolveEulerIntegrator,
    MCSolveIntegrator,
    MCSolveKvaerno3Integrator,
    MCSolveKvaerno5Integrator,
    MCSolveTsit5Integrator,
)

__all__ = ['mcsolve']


def mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    keys: list[PRNGKey] = [PRNGKey(42)],
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    root_finder: AbstractRootFinder = optx.Newton(1e-5, 1e-5, optx.rms_norm),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MCSolveResult:
    r"""Perform Monte-Carlo evolution, unraveling the master equation.

    We follow the algorithm outlined in Abdelhafez et al. to efficiently perform
    Monte-Carlo sampling. First the no-jump trajectory is computed for a state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi_0}$, according to the Schrödinger
    equation with non-Hermitian Hamiltonian ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt} = -i (H(t)
            -i/2 \sum_{k=1}^{N}L_{k}^{\dag}(t)L_{k}(t) ) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators at time $t$. We then extract the norm of the state
    at the final time, and take this as a lower bound for random numbers sampled
    for other trajectories such that they all experience at least one jump.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.constant()`](/python_api/time_array/constant.html),
        [`dq.pwc()`](/python_api/time_array/pwc.html),
        [`dq.modulated()`](/python_api/time_array/modulated.html), or
        [`dq.timecallable()`](/python_api/time_array/timecallable.html).

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `jump_ops` and the
         initial state `psi0` can be batched to solve multiple monte-carlo equations concurrently.
        All other arguments are common to every batch.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        jump_ops _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        psi0 _(array-like of shape (bpsi?, n, 1))_: Initial state.
        tsave _(array-like of shape (nt,))_: Times at which the states and expectation
            values are saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or
            from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        ntraj _(int, optional)_: Total number of trajectories to simulate, including
            the no-jump trajectory. Defaults to 10.
        key _(PRNGKeyArray, optional)_: random key to use for monte-carlo sampling.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5()`](/python_api/solver/Tsit5.html).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`](/python_api/options/Options.html).

    Returns:
        [`dq.Result`](/python_api/result/Result.html) object holding the result of the
            Monte-Carlo integration. It has the following attributes:

            - **states** _(array of shape (bH?, brho?, ntraj, nt, n, n))_ -- Saved states.
            - **expects** _(array of shape (bH?, brho?, nE, nt), optional)_ -- Saved
                expectation values.
            - **extra** _(PyTree, optional)_ -- Extra data saved with `save_extra()` if
                specified in `options`.
            - **infos** _(PyTree, optional)_ -- Solver-dependent information on the
                resolution.
            - **tsave** _(array of shape (nt,))_ -- Times for which states and
                expectation values were saved.
            - **solver** _(Solver)_ -- Solver used.
            - **gradient** _(Gradient)_ -- Gradient used.
            - **options** _(Options)_ -- Options used.
    """
    # === convert arguments
    H = _astimearray(H)
    jump_ops = [_astimearray(L) for L in jump_ops]
    psi0 = jnp.asarray(psi0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_mcsolve_args(H, jump_ops, psi0, tsave, exp_ops)

    return _vectorized_mcsolve(
        H, jump_ops, psi0, tsave, keys, exp_ops, solver, root_finder, gradient, options
    )


def _vectorized_mcsolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    keys: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCSolveResult:
    # === vectorize function
    # we vectorize over H, jump_ops, and psi0. keys are vectorized over inside of run().

    out_axes = MCSolveResult(False, False, False, False, 0, 0, 0, 0, 0, False)

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(
            H.shape[:-2], psi0.shape[:-2], *[jump_op.shape[:-2] for jump_op in jump_ops]
        )

        def broadcast(x: TimeArray) -> TimeArray:
            return x.broadcast_to(*(broadcast_shape + x.shape[-2:]))

        H = broadcast(H)
        jump_ops = list(map(broadcast, jump_ops))
        psi0 = jnp.broadcast_to(psi0, broadcast_shape + psi0.shape[-2:])

    n_batch = (
        H.in_axes,
        [jump_op.in_axes for jump_op in jump_ops],
        Shape(psi0.shape[:-2]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_mcsolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_mcsolve, n_batch, out_axes)

    # === apply vectorized function
    return f(
        H, jump_ops, psi0, tsave, keys, exp_ops, solver, root_finder, gradient, options
    )


@partial(jax.jit, static_argnames=['solver', 'root_finder', 'gradient', 'options'])
def _mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    keys: list[PRNGKey],
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCSolveResult:
    integrators = {
        Euler: MCSolveEulerIntegrator,
        Dopri5: MCSolveDopri5Integrator,
        Dopri8: MCSolveDopri8Integrator,
        Tsit5: MCSolveTsit5Integrator,
        Kvaerno3: MCSolveKvaerno3Integrator,
        Kvaerno5: MCSolveKvaerno5Integrator,
    }
    integrator_class: MCSolveIntegrator = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_class(
        ts=tsave,
        y0=psi0,
        solver=solver,
        gradient=gradient,
        options=options,
        H=H,
        Ls=jump_ops,
        keys=keys,
        root_finder=root_finder,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mcsolve_args(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    exp_ops: Array | None,
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check tsave shape
    check_times(tsave, 'tsave')

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'len(exp_ops)'})
