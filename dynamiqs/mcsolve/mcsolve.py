from __future__ import annotations

import logging
from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox.internal import while_loop
from jax.random import PRNGKey
from jax import Array
from jaxtyping import ArrayLike
from optimistix import AbstractRootFinder

from .._checks import check_shape, check_times
from .._utils import cdtype
from ..core._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_solver_class,
)
from ..gradient import Gradient
from ..options import Options
from ..result import Result, MCResult
from ..solver import Dopri5, Dopri8, Euler, Solver, Tsit5
from ..time_array import TimeArray, Shape
from ..utils.utils import unit, dag
from .mcdiffrax import MCDopri5, MCDopri8, MCEuler, MCTsit5

__all__ = ["mcsolve"]


def mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    key: PRNGKey = PRNGKey(42),
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    root_finder: AbstractRootFinder = optx.Newton(1e-5, 1e-5, optx.rms_norm),
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> Result:
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
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    # === check arguments
    _check_mcsolve_args(H, jump_ops, psi0, tsave, exp_ops)

    return _vectorized_mcsolve(
        H, jump_ops, psi0, tsave, key, exp_ops, solver, root_finder, gradient, options
    )


@catch_xla_runtime_error
def _vectorized_mcsolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCResult:
    # === vectorize function
    # we vectorize over H, jump_ops and psi0, all other arguments are not vectorized
    # below we will have another layer of vectorization over ntraj

    out_axes = MCResult(None, 0, 0, 0, 0)

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
    # the result is vectorized over `saved`

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_mcsolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_mcsolve, n_batch, out_axes)

    # === apply vectorized function
    return f(H, jump_ops, psi0, tsave, key, exp_ops, solver, root_finder, gradient, options)


def _mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCResult:
    ntraj = options.ntraj
    key_1, key_2, key_3 = jax.random.split(key, num=3)
    # simulate no-jump trajectory
    rand0 = 0.0
    no_jump_result = _single_traj(
        H, jump_ops, psi0, tsave, rand0, exp_ops, solver, root_finder, gradient, options
    )
    # extract the no-jump probability
    no_jump_state = no_jump_result.final_state
    p_nojump = jnp.abs(jnp.einsum("id,id->", jnp.conj(no_jump_state), no_jump_state))
    # TODO split earlier so that not reusing key for different batch dimensions
    random_numbers = jax.random.uniform(key_2, shape=(ntraj,), minval=p_nojump)
    # run all single trajectories at once
    traj_keys = jax.random.split(key_3, num=ntraj)
    if options.one_jump_only:
        f = jax.vmap(
            one_jump_only,
            in_axes=(None, None, None, None, 0, 0, None, None, None, None, None),
        )
    else:
        f = jax.vmap(
            loop_over_jumps,
            in_axes=(None, None, None, None, 0, 0, None, None, None, None, None),
        )
    jump_results = f(
        H,
        jump_ops,
        psi0,
        tsave,
        traj_keys,
        random_numbers,
        exp_ops,
        solver,
        root_finder,
        gradient,
        options,
    )
    if no_jump_result.expects is not None:
        jump_expects = jnp.mean(jump_results.expects, axis=0)
        no_jump_expects = no_jump_result.expects
        avg_expects = p_nojump * no_jump_expects + (1 - p_nojump) * jump_expects
    else:
        avg_expects = None
    mcresult = MCResult(tsave, no_jump_result, jump_results, p_nojump, avg_expects)
    return mcresult


@partial(jax.jit, static_argnames=('solver', 'root_finder', 'gradient', 'options'))
def _single_traj(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    rand: Array,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
):
    """function that actually performs the time evolution"""
    solvers = {
        Euler: MCEuler,
        Dopri5: MCDopri5,
        Dopri8: MCDopri8,
        Tsit5: MCTsit5,
    }
    solver_class = get_solver_class(solvers, solver)
    solver.assert_supports_gradient(gradient)
    mcsolver = solver_class(
        tsave, psi0, H, exp_ops, solver, gradient, options, jump_ops, rand, root_finder,
    )
    return mcsolver.run()


def one_jump_only(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    rand: float,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
):
    key_1, key_2 = jax.random.split(key)
    before_jump_result = _jump_trajs(
        H, jump_ops, psi0, tsave, key_1, rand, exp_ops, solver, root_finder, gradient, options
    )
    new_t0 = before_jump_result.final_time
    new_psi0 = before_jump_result.final_state
    new_tsave = jnp.linspace(new_t0, tsave[-1], len(tsave))
    # don't allow another jump
    after_jump_result = _jump_trajs(
        H, jump_ops, new_psi0, new_tsave, key_2, 0.0, exp_ops, solver, root_finder, gradient, options
    )
    result = interpolate_states_and_expects(
        tsave, new_tsave, before_jump_result, after_jump_result, new_t0, options
    )
    return result


def loop_over_jumps(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    rand: float,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
):
    """loop over jumps until the simulation reaches the final time"""
    def while_cond(t_state_key_solver):
        prev_result, prev_key = t_state_key_solver
        return prev_result.final_time < tsave[-1]

    def while_body(t_state_key_solver):
        prev_result, prev_key = t_state_key_solver
        jump_key, next_key, loop_key = jax.random.split(prev_key, num=3)
        new_rand = jax.random.uniform(jump_key)
        new_t0 = prev_result.final_time
        new_psi0 = prev_result.final_state
        # tsave_after_jump has spacings not consistent with tsave, but
        # we will interpolate later
        new_tsave = jnp.linspace(new_t0, tsave[-1], len(tsave))
        next_result = _jump_trajs(
            H,
            jump_ops,
            new_psi0,
            new_tsave,
            next_key,
            new_rand,
            exp_ops,
            solver,
            root_finder,
            gradient,
            options,
        )
        result = interpolate_states_and_expects(
            tsave, new_tsave, prev_result, next_result, new_t0, options
        )
        return result, loop_key

    # solve until the first jump occurs. Enter the while loop for additional jumps
    key_1, key_2 = jax.random.split(key)
    initial_result = _jump_trajs(
        H, jump_ops, psi0, tsave, key_1, rand, exp_ops, solver, root_finder, gradient, options
    )

    final_result, _ = while_loop(
        while_cond,
        while_body,
        (initial_result, key_2),
        max_steps=100,
        kind="checkpointed",
    )
    return final_result


def _jump_trajs(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    rand: float,
    exp_ops: Array | None,
    solver: Solver,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
):
    """for the jump trajectories, call _single_traj and then apply a
    jump if t < tsave[-1]"""
    rand_key, sample_key = jax.random.split(key)
    # solve until jump or tsave[-1]
    res_before_jump = _single_traj(
        H, jump_ops, psi0, tsave, rand, exp_ops, solver, root_finder, gradient, options
    )
    t_jump = res_before_jump.final_time
    psi_before_jump = res_before_jump.final_state
    # select and apply a random jump operator, renormalize
    jump_op = sample_jump_ops(t_jump, psi_before_jump, jump_ops, sample_key)
    # if t = tsave[-1], this is reason for solver termination (no jump)
    psi = jnp.where(
        t_jump < tsave[-1], unit(jump_op @ psi_before_jump), psi_before_jump
    )
    # save this state as the new final state
    result = eqx.tree_at(lambda res: res._saved.ylast, res_before_jump, psi)
    return result


def interpolate_states_and_expects(
    tsave, tsave_after_jump, results_before_jump, results_after_jump, t_jump, options
):
    """create a cubic spline of the states and expectation values from after a jump. The tsave
    used after a jump likely does not correspond in terms of spacing and knots to the tsave from
    before the jump, so we use this spline to obtain the states and expectation values at
    the requested locations."""
    if options.save_states:
        states = _interpolate_intermediate_results(
            tsave,
            tsave_after_jump,
            results_before_jump.states,
            results_after_jump.states,
            t_jump,
            expand_axis_dims=(1, 2),
        )
        results_after_jump = eqx.tree_at(
            lambda res: res._saved.ysave, results_after_jump, states
        )
    if results_before_jump.expects is not None:
        before_expects = results_before_jump.expects.swapaxes(
            -1, -2
        )  # (nE, nt) -> (nt, nE)
        after_expects = results_after_jump.expects.swapaxes(-1, -2)
        expects = _interpolate_intermediate_results(
            tsave,
            tsave_after_jump,
            before_expects,
            after_expects,
            t_jump,
            expand_axis_dims=(1,),
        )
        results_after_jump = eqx.tree_at(
            lambda res: res._saved.Esave, results_after_jump, expects.swapaxes(-1, -2)
        )
    return results_after_jump


def _interpolate_intermediate_results(
    original_tsave,
    tsave_after_jump,
    results_before_jump,
    results_after_jump,
    t_jump,
    expand_axis_dims=(1, 2),
):
    # replace infs with nans to avoid cubic splining to infinity
    results_after_jump_nan = jnp.where(
        jnp.isinf(results_after_jump), jnp.nan, results_after_jump
    )
    coeffs = dx.backward_hermite_coefficients(tsave_after_jump, results_after_jump_nan)
    cubic_spline = dx.CubicInterpolation(tsave_after_jump, coeffs)
    f = jax.vmap(cubic_spline.evaluate, in_axes=(0,))
    # this spline will be evaluated outside of the range it was fitted in (before t_jump). No
    # matter, that bad data is replaced by data from results_before_jump
    post_jump_values = f(original_tsave)
    mask = jnp.expand_dims(original_tsave < t_jump, axis=expand_axis_dims)
    results = jnp.where(mask, results_before_jump, post_jump_values)
    return results


def sample_jump_ops(t, psi, jump_ops, key, eps=1e-15):
    """given a state psi at time t that should experience a jump,
    randomly sample one jump operator from among the provided jump_ops.
    The probability that a certain jump operator is selected is weighted
    by the probability that such a jump can occur. For instance for a qubit
    experiencing amplitude damping, if it is in the ground state then
    there is probability zero of experiencing an amplitude damping event.
    """
    Ls = jnp.stack([L(t) for L in jump_ops])
    Lsd = dag(Ls)
    # i, j, k: hilbert dim indices; e: jump ops; d: index of dimension 1
    probs = jnp.einsum("id,eij,ejk,kd->e", jnp.conj(psi), Lsd, Ls, psi)
    # for categorical we pass in the log of the probabilities
    logits = jnp.log(jnp.real(probs / (jnp.sum(probs) + eps)))
    # randomly sample the index of a single jump operator
    sample_idx = jax.random.categorical(key, logits, shape=(1,))
    # extract that jump operator and squeeze size 1 dims
    return jnp.squeeze(jnp.take(Ls, sample_idx, axis=0), axis=0)


def _check_mcsolve_args(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    exp_ops: Array | None,
):
    # === check H shape
    check_shape(H, 'H', '(?, n, n)', subs={'?': 'nH?'})

    # === check jump_ops shape
    if len(jump_ops) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(?, n, n)', subs={'?': 'nL?'})

    # === check rho0 shape
    check_shape(psi0, 'rho0', '(?, n, 1)', subs={'?': 'npsi0?'})

    # === check tsave shape
    check_times(tsave, 'tsave')

    # === check exp_ops shape
    if exp_ops is not None:
        check_shape(exp_ops, 'exp_ops', '(N, n, n)', subs={'N': 'nE'})
