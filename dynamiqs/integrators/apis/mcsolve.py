from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jaxtyping import ArrayLike
from optimistix import AbstractRootFinder

from ..._checks import check_shape, check_times
from ..._utils import cdtype
from ...gradient import Gradient
from ...options import Options
from ...result import MCSolveResult
from ...solver import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import TimeArray
from .._utils import _astimearray, cartesian_vmap, get_integrator_class, multi_vmap, catch_xla_runtime_error
from ..mcsolve.diffrax_integrator import (
    MCSolveDopri5Integrator,
    MCSolveDopri8Integrator,
    MCSolveEulerIntegrator,
    MCSolveIntegrator,
    MCSolveKvaerno3Integrator,
    MCSolveKvaerno5Integrator,
    MCSolveTsit5Integrator,
)


def mcsolve(
    H: ArrayLike | TimeArray,
    Ls: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    keys: ArrayLike = jax.random.split(jax.random.key(31), num=10),  # noqa: B008
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    root_finder: AbstractRootFinder | None = optx.Newton(1e-5, 1e-5, optx.rms_norm),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MCSolveResult:
    r"""Perform Monte-Carlo evolution, unraveling the master equation.

    We follow the algorithm outlined in Abdelhafez et al. to efficiently perform
    Monte-Carlo sampling. First the no-jump trajectory is computed for a state vector
    $\ket{\psi(t)}$ at time $t$, starting from an initial state $\ket{\psi_0}$,
    according to the Schrödinger equation with non-Hermitian Hamiltonian ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt}
        = -i [H(t) -\frac{i}{2} \sum_{k=1}^{N} L_{k}^{\dagger} (t)L_{k}(t)] \ket{\psi(t)},
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
        The Hamiltonian `H`, the jump operators `Ls` and the
        initial state `psi0` can be batched to solve multiple monte-carlo equations
        concurrently. All other arguments are common to every batch.

    Args:
        H _(array-like or time-array of shape (bH?, n, n))_: Hamiltonian.
        Ls _(list of array-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        psi0 _(array-like of shape (bpsi?, n, 1))_: Initial state.
        tsave _(array-like of shape (nt,))_: Times at which the states and expectation
            values are saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or
            from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        keys _(KeyArray of shape (ntraj,))_: Total number of jump trajectories to
            simulate, not including the no-jump trajectory. Defaults to a list of keys
            of length 10.
        exp_ops _(list of array-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5()`](/python_api/solver/Tsit5.html).
        root_finder: Root finder passed to dx.diffeqsolve() to find the exact time an
            event occurs. Can be `None`, in which case the root finding functionality
            is not utilized. It is recommended to pass a root finder (such as the
            default Newton root finder) so that event times are not determined by the
            integration step sizes in diffeqsolve. However there are cases where the
            root finding can fail, causing the whole simulation to fail. Passing `None`
            for `root_finder` can alleviate the issue in these cases.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`](/python_api/options/Options.html).

    Returns:
        [`dq.Result`](/python_api/result/Result.html) object holding the result of the
            Monte-Carlo integration. It has the following attributes:

            - **no_jump_states** _(array of shape (bH?, bpsi0?, nt, n, 1))_ -- Saved
                no-jump states.
            - **final_no_jump_state** _(array of shape (bH?, bpsi0?, n, 1))_ -- Saved
                final no-jump state.
            - **jump_states** _(array of shape (bH?, bpsi0?, ntraj, nt, n, 1))_ -- Saved
                jump states.
            - **final_jump_states** _(array of shape (bH?, bpsi0?, ntraj, n, 1))_ --
                Saved final jump states.
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
    """  # noqa E501
    # === convert arguments
    H = _astimearray(H)
    Ls = [_astimearray(L) for L in Ls]
    psi0 = jnp.asarray(psi0, dtype=cdtype())
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_mcsolve_args(H, Ls, psi0, tsave, exp_ops)

    return _vectorized_mcsolve(
        H, Ls, psi0, tsave, keys, exp_ops, solver, root_finder, gradient, options
    )


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=['solver', 'root_finder', 'gradient', 'options'])
def _vectorized_mcsolve(
    H: TimeArray,
    Ls: list[TimeArray],
    psi0: Array,
    tsave: Array,
    keys: Array,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCSolveResult:
    # === vectorize function
    # vectorize output over `_no_jump_res`, `_jump_res`, `no_jump_prob`, `jump_times`,
    # `num_jumps`
    out_axes = MCSolveResult(None, None, None, None, 0, 0, 0, 0, 0)

    if options.cartesian_batching:
        # vectorize input over H, Ls, psi0.
        in_axes = (
            H.in_axes,
            [L.in_axes for L in Ls],
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        nvmap = (
            H.ndim - 2,
            [L.ndim - 2 for L in Ls],
            psi0.ndim - 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        f = cartesian_vmap(_mcsolve, in_axes, out_axes, nvmap)
    else:
        # vectorize input over H, Ls, psi0 and keys.
        in_axes = (
            H.in_axes,
            [L.in_axes for L in Ls],
            0,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
        )
        shapes = [x.shape[:-2] for x in [H, *Ls, psi0, keys]]
        bshape = jnp.broadcast_shapes(*shapes)
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        psi0 = jnp.broadcast_to(psi0, (*bshape, n, 1))
        keys = jnp.broadcast_to(keys, (*bshape, *keys.shape[-2:]))
        # vectorize the function
        f = multi_vmap(_mcsolve, in_axes, out_axes, nvmap)

    return f(H, Ls, psi0, tsave, keys, exp_ops, solver, root_finder, gradient, options)


def _mcsolve(
    H: ArrayLike | TimeArray,
    Ls: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    keys: Array,
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
        Ls=Ls,
        keys=keys,
        root_finder=root_finder,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mcsolve_args(
    H: TimeArray, Ls: list[TimeArray], psi0: Array, tsave: Array, exp_ops: Array | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'Ls[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        logging.warning(
            'Argument `Ls` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check tsave shape
    check_times(tsave, 'tsave')

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'len(exp_ops)'})
