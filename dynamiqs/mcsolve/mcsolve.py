from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax import Array
from jaxtyping import ArrayLike

from .._utils import cdtype
from ..core._utils import _astimearray, compute_vmap, get_solver_class
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5
from ..time_array import TimeArray
from ..utils.utils import todm, norm, unit, dag
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


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
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
    # we vectorize over H, jump_ops and rho0, all other arguments are not vectorized
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
    out_axes = Result(None, None, None, None, 0, 0)
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
) -> Result:
    # simulate no-jump trajectory
    # run the no jump
    no_jump_state = _single_traj(H, jump_ops, psi0, tsave, key, jnp.zeros(shape=(1, 1)), exp_ops, solver, gradient, options)
    # extract the no-jump probability, which sets a lower bound on the random
    # numbers we sample
    norm_nojump = norm(no_jump_state)
    # split key into ntraj keys
    # TODO split earlier so that not reusing key for different batch dimensions
    keys = jax.random.split(key, ntraj)
    random_numbers = jax.random.uniform(keys[0], shape=(ntraj,), minval=norm_nojump)
    # run all single trajectories at once
    # 0 indicates the dimension to vmap over. Here that is the random numbers along
    # with their keys, which come along for the ride so that we can draw further random
    # numbers for which jumps to apply
    run_trajs = jax.vmap(_single_traj, in_axes=(None, None, None, None, 0, 0, None, None, None, None))
    psis = run_trajs(H, jump_ops, psi0, tsave, keys, random_numbers, exp_ops, solver, gradient, options)
    return no_jump_state, psis, norm_nojump


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
        t, state, key, solver = t_state_key_solver
        return t < tsave[-1]

    def while_body(t_state_key_solver):
        t, state, key, solver = t_state_key_solver
        solvers = {
            Euler: MCEuler,
            Dopri5: MCDopri5,
            Dopri8: MCDopri8,
            Tsit5: MCTsit5,
        }
        solver_class = get_solver_class(solvers, solver)
        solver.assert_supports_gradient(gradient)
        # new_t0_idx = jnp.argmin(tsave - t)
        # TODO fix so that we compute exp_ops appropriately
        # new_tsave = jnp.concat((t, tsave[new_t0_idx:]))
        new_tsave = jnp.linspace(t, tsave[-1], 2)
        solver = solver_class(new_tsave, state, H, exp_ops, solver, gradient, options, jump_ops)
        # solve until jump
        res = solver.run()
        t_jump = res.times[-1]
        state_before_jump = res.states[-1]
        psi_before_jump = state_before_jump[0:-1]

        # apply jump and renormalize
        key, sample_key = jax.random.split(key)
        jump_op = sample_jump_ops(t_jump, psi_before_jump, jump_ops, sample_key)
        # only apply a jump if the solve terminated before the end
        mask = jnp.where(
            t_jump < tsave[-1],
            if_true=jnp.array([1.0,]),
            if_false=jnp.array([0.0,])
        )
        psi = unit(mask * jump_op @ psi_before_jump + (1 - mask) * psi_before_jump)
        # generate new random number
        key, new_key = jax.random.split(key)
        r = jax.random.uniform(key)
        state = jnp.concat((psi, r))
        return t_jump, state, new_key, solver

    initial_state = jnp.concat((psi0, rand))
    _, state, _, _ = jax.lax.while_loop(while_cond, while_body, (tsave[0], initial_state, key, solver))
    return state


def sample_jump_ops(t, psi, jump_ops, key):
    Ls = jnp.stack([L(t) for L in jump_ops])
    Lsd = dag(Ls)
    probs = jnp.einsum("i,eij,ejk,k->e",
                       jnp.conj(psi), Lsd, Ls, psi
                       )
    logits = jnp.log(probs)
    sample_idx = jax.random.categorical(key, logits, shape=(1,))
    return jump_ops[sample_idx[0]]
