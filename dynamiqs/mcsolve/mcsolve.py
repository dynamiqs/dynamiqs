from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax.random import PRNGKey
from jax import Array
from jaxtyping import ArrayLike

from .._utils import cdtype
from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result, Saved, MCResult, FinalSaved
from ..solver import Dopri5, Dopri8, Euler, Solver, Tsit5
from ..time_array import TimeArray
from ..utils.utils import unit, dag
from .mcdiffrax import MCDopri5, MCDopri8, MCEuler, MCTsit5

__all__ = ['mcsolve']


def mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    ntraj: int = 10,
    key: PRNGKey = PRNGKey(42),
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
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
    return _vmap_mcsolve(H, jump_ops, psi0, tsave, ntraj, key, exp_ops, solver, gradient, options)


def _vmap_mcsolve(
    H: TimeArray,
    jump_ops: list[TimeArray],
    psi0: Array,
    tsave: Array,
    ntraj: int,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> Result:
    # === vectorize function
    # we vectorize over H, jump_ops and psi0, all other arguments are not vectorized
    # below we will have another layer of vectorization over ntraj
    is_batched = (
        H.ndim > 2,
        [jump_op.ndim > 2 for jump_op in jump_ops],
        psi0.ndim > 2,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    # the result is vectorized over `saved`
    out_axes = MCResult(None, 0, 0, 0)
    f = compute_vmap(_mcsolve, options.cartesian_batching, is_batched, out_axes)
    return f(H, jump_ops, psi0, tsave, ntraj, key, exp_ops, solver, gradient, options)


def _mcsolve(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    ntraj: int,
    key: PRNGKey,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> MCResult:
    key_1, key_2, key_3 = jax.random.split(key, num=3)
    # simulate no-jump trajectory
    rand0 = jnp.zeros(shape=(1, 1))
    no_jump_result = _single_traj(H, jump_ops, psi0, tsave, key_1, rand0, exp_ops, solver, gradient, options)
    # extract the no-jump probability
    # TODO want to keep the rands around in the Result containers?
    no_jump_state_sq = jnp.squeeze(no_jump_result.final_state[0:-1])
    p_nojump = jnp.conj(no_jump_state_sq) @ no_jump_state_sq
    # split key into ntraj keys
    # TODO split earlier so that not reusing key for different batch dimensions
    random_numbers = jax.random.uniform(key_2, shape=(ntraj, 1, 1), minval=p_nojump)
    # run all single trajectories at once
    # 0 indicates the dimension to vmap over. Here that is the random numbers along
    # with their keys, which come along for the ride so that we can draw further random
    # numbers for which jumps to apply
    traj_keys = jax.random.split(key_3, num=ntraj)
    run_trajs = jax.vmap(_single_traj, in_axes=(None, None, None, None, 0, 0, None, None, None, None),)
    jump_results = run_trajs(H, jump_ops, psi0, tsave, traj_keys, random_numbers, exp_ops, solver, gradient, options)
    mcresult = MCResult(tsave, no_jump_result, jump_results, p_nojump)
    return mcresult


#@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _single_traj(
    H: ArrayLike | TimeArray,
    jump_ops: list[ArrayLike | TimeArray],
    psi0: ArrayLike,
    tsave: ArrayLike,
    key: PRNGKey,
    rand: float,
    exp_ops: Array | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
):
    def while_cond(t_state_key_solver):
        t, prev_tsave, prev_key, prev_solver, prev_results = t_state_key_solver
        return t < tsave[-1]

    def while_body(t_state_key_solver):
        t, prev_tsave, prev_key, prev_solver, prev_result = t_state_key_solver
        next_r_key, sample_key, next_loop_key = jax.random.split(prev_key, num=3)
        solvers = {
            Euler: MCEuler,
            Dopri5: MCDopri5,
            Dopri8: MCDopri8,
            Tsit5: MCTsit5,
        }
        solver_class = get_solver_class(solvers, prev_solver)
        prev_solver.assert_supports_gradient(gradient)
        prev_state = prev_result.final_state
        mcsolver = solver_class(prev_tsave, prev_state, H, exp_ops, prev_solver, gradient, options, jump_ops)
        # solve until jump
        new_result = mcsolver.run()
        # in the next round want to keep saving at the same instances in time, not
        # affected by the specific time a jump occurs
        t_jump = new_result.final_time
        # want to use OG tsave here
        new_start_index = jnp.searchsorted(tsave, t_jump, side="right")
        # start saving at the next time after a jump
        new_tsave = jnp.where(tsave < t_jump, jnp.inf, tsave)
        concatenated_res = add_results(prev_result, new_result, new_start_index, options)
        state_before_jump = new_result.final_state
        # peel off the state, excluding the random number
        psi_before_jump = state_before_jump[0:-1]
        # randomly sample one jump operator
        jump_op = sample_jump_ops(t_jump, psi_before_jump, jump_ops, sample_key)
        # only apply a jump if the solve terminated before the end
        mask = jnp.where(t_jump < tsave[-1], jnp.array([1.0, ]), jnp.array([0.0, ]))
        # renormalize the state if a jump is applied
        psi = unit(mask * jump_op @ psi_before_jump) + (1 - mask) * psi_before_jump
        # new random key for the next round of the while loop
        r = jax.random.uniform(next_r_key, shape=(1, 1))
        new_state = jnp.concatenate((psi, r))
        concatenated_res = eqx.tree_at(
            lambda res: res.final_state, concatenated_res, new_state
        )
        concatenated_res = eqx.tree_at(
            lambda res: res.infos, concatenated_res, None
        )
        return t_jump, new_tsave, next_loop_key, prev_solver, concatenated_res

    initial_state = jnp.concatenate((psi0, rand))
    # initialize arrays for future concatenation
    if options.save_states:
        n = initial_state.shape[0]
        nT = tsave.shape[0]
        ysave = jnp.full(shape=(nT, n, 1), fill_value=jnp.inf, dtype=cdtype())
    else:
        ysave = initial_state
    if exp_ops is not None and len(exp_ops) > 0:
        nE = len(exp_ops)
        nT = tsave.shape[0]
        Esave = jnp.full(shape=(nT, nE), fill_value=jnp.inf, dtype=cdtype())
    else:
        Esave = None
    saved = FinalSaved(ysave, Esave, None, initial_state)
    initial_result = Result(
        tsave, solver, gradient, options, saved, tsave[0], None,
    )

    _, _, _, _, final_results = while_loop(
        while_cond, while_body, (tsave[0], tsave, key, solver, initial_result),
        kind="checkpointed", max_steps=100
    )
    return final_results


def sample_jump_ops(t, psi, jump_ops, key, eps=1e-15):
    Ls = jnp.stack([L(t) for L in jump_ops])
    Lsd = dag(Ls)
    # i, j, k: hilbert dim indices; e: jump ops; d: index of dimension 1
    probs = jnp.einsum("id,eij,ejk,kd->e",
                       jnp.conj(psi), Lsd, Ls, psi
                       )
    logits = jnp.log(jnp.real(probs / (jnp.sum(probs)+eps)))
    # randomly sample the index of a single jump operator
    sample_idx = jax.random.categorical(key, logits, shape=(1,))
    # extract that jump operator and squeeze size 1 dims
    return jnp.squeeze(jnp.take(Ls, sample_idx, axis=0), axis=0)


def add_results(old_res, new_res, new_start_index, options):
    """ add results from two different runs, where old_res results from
     a simulation that experienced a jump. diffrax convention is that
     all result arrays are filled with jnp.infs for those values beyond
     the time where a jump occured.
     """
    if options.save_states:
        concatenated_states = _add_inf_arrays(
            old_res.states, new_res.states, new_start_index, axis=-3
        )
        # worried about accessing protected attribute...
        new_res = eqx.tree_at(lambda res: res._saved.ysave, new_res, concatenated_states)
    if old_res.expects is not None:
        concatenated_expects = _add_inf_arrays(
            old_res.expects, new_res.expects, new_start_index, axis=-2
        )
        new_res = eqx.tree_at(lambda res: res._saved.Esave, new_res, concatenated_expects)
    return new_res


def _add_inf_arrays(old_array_with_infs, replacement_array, t_idx, axis=-3):
    inf_mask = jnp.isinf(old_array_with_infs)
    # for states roll along the third-to-last axis, as states has shape ... nt, n, 1
    # for expects roll along the second-to-last axis, as expects has shape ... nt, nE
    rolled_new_states = jnp.roll(replacement_array, shift=t_idx, axis=axis)
    new_array = jnp.where(inf_mask, rolled_new_states, old_array_with_infs)
    return new_array
