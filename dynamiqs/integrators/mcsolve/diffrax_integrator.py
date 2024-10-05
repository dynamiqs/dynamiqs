from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
import jax
import equinox as eqx
from jax.random import PRNGKey
import jax.tree_util as jtu
from equinox.internal import while_loop

import warnings

from jax import Array
from jaxtyping import PyTree

from ..core.save_mixin import SolveSaveMixin
from ...utils.quantum_utils import dag, unit, norm
from ..core.abstract_integrator import MCSolveIntegrator
from ..core.diffrax_integrator import (
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    MCDiffraxIntegrator,
    Tsit5Integrator,
)

MAX_JUMPS = 100


class SaveState(eqx.Module):
    ts: Array
    ys: PyTree
    save_index: int
    jump_times: Array


class State(eqx.Module):
    save_state: PyTree[SaveState]
    final_state: Array
    final_time: float
    num_jumps: int
    key: PRNGKey


def _inner_buffers(save_state):
    assert type(save_state) is SaveState
    return save_state.ts, save_state.ys, save_state.jump_times


def _outer_buffers(state):
    assert type(state) is State
    is_save_state = lambda x: isinstance(x, SaveState)
    # state.save_state has type PyTree[SaveState]. In particular this may include some
    # `None`s, which may sometimes be treated as leaves (e.g.
    # `tree_at(_outer_buffers, ..., is_leaf=lambda x: x is None)`).
    # So we need to only get those leaves which really are a SaveState.
    save_states = jtu.tree_leaves(state.save_state, is_leaf=is_save_state)
    save_states = [x for x in save_states if is_save_state(x)]
    return (
        [s.ts for s in save_states]
        + [s.ys for s in save_states]
        + [s.jump_times for s in save_states]
    )


class MCSolveDiffraxIntegrator(MCDiffraxIntegrator, MCSolveIntegrator, SolveSaveMixin):
    """Integrator computing the time evolution of the Monte-Carlo unraveling of the
    Lindblad master equation using the Diffrax library."""

    def _run(self, y0, rand):
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)
            # TODO: remove once https://github.com/patrick-kidger/diffrax/issues/445 is
            # closed
            warnings.simplefilter('ignore', FutureWarning)

            saveat, adjoint = self.diffrax_arguments()

            def cond_fn(t, y, *args, **kwargs):
                norm = jnp.linalg.norm(y)
                return norm**2 - rand

            event = dx.Event(cond_fn, self.root_finder)
            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                event=event,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )
        return solution

    def run(self):
        no_jump_solution = self._run(self.y0, 0.0)
        # don't need ellipses preceding 0 since this has been previously vmapped
        final_no_jump_state = no_jump_solution.ys[1][0, :, :]
        no_jump_prob = norm(final_no_jump_state) ** 2
        # set up jump and dark state vmapped functions
        jump_state_fun = jax.vmap(self._loop_over_jumps, in_axes=(0, None))
        dark_state_fun = jax.vmap(self._dark_state, in_axes=(0, None))
        # call jump_state_fun only if no_jump_prob is below 0.999
        jump_state = jax.lax.cond(
            1 - no_jump_prob > 1e-3,
            jump_state_fun,
            dark_state_fun,
            *(self.keys, no_jump_prob),
        )
        jump_times = jump_state.save_state.jump_times
        num_jumps = jump_state.num_jumps
        # === save and postprocess results
        no_jump_saved = self.postprocess_saved(*no_jump_solution.ys)
        no_jump_result = self.traj_result(
            no_jump_saved, infos=self.infos(no_jump_solution.stats)
        )

        jump_saved = self.postprocess_saved(
            jump_state.save_state.ys, jump_state.final_state
        )
        # TODO save stats for jumps
        jump_result = self.traj_result(jump_saved)
        return self.result(
            no_jump_result, jump_result, no_jump_prob, jump_times, num_jumps
        )

    def _dark_state(self, key, no_jump_prob):
        result = self._run(self.y0, 0.0)
        save_state = SaveState(
            ts=result.ts[0],
            ys=result.ys[0],
            save_index=len(self.ts),
            jump_times=jnp.full(MAX_JUMPS, jnp.nan),
        )
        state = State(
            save_state=save_state,
            final_state=result.ys[1][0, :, :],
            final_time=result.ts[1],
            num_jumps=0,
            key=key,
        )

    def _loop_over_jumps(self, key, no_jump_prob):
        """loop over jumps until the simulation reaches the final time"""
        rand_key, sample_key = jax.random.split(key)
        rand = jax.random.uniform(rand_key, minval=no_jump_prob)
        first_result = self._run(self.y0, rand)
        first_jump_time = first_result.ts[1]
        time_diffs = self.ts - first_jump_time
        # want to start with a time after the jump
        time_diffs = jnp.where(time_diffs < 0, jnp.inf, time_diffs)
        save_index = jnp.argmin(time_diffs)
        # allocate memory for the state that is updated during the loop
        save_state = SaveState(
            ts=first_result.ts[0],
            ys=first_result.ys[0],
            save_index=save_index,
            jump_times=jnp.full(MAX_JUMPS, jnp.nan),
        )
        state = State(
            save_state=save_state,
            final_state=first_result.ys[1][0, :, :],
            final_time=first_jump_time,
            num_jumps=0,
            key=sample_key,
        )

        def outer_cond_fun(_state):
            return _state.final_time < self.t1

        def outer_body_fun(_state):
            psi = _state.final_state
            jump_time = _state.final_time
            # if we've (re)entered the loop, that means we terminated before the final time
            # so we need a jump
            # select and apply a random jump operator, renormalize
            _next_key, _rand_key, _sample_key = jax.random.split(_state.key, num=3)
            jump_op = self.sample_jump_ops(jump_time, psi, _sample_key)
            # don't need conditional here on whether to apply jump?
            psi_after_jump = unit(jump_op @ psi)
            # new linspace over the remaining times
            new_times = jnp.linspace(jump_time, self.t1, len(self.ts))
            # run the next leg until it either terminates for a new jump or it finishes
            _rand = jax.random.uniform(_rand_key)
            result_after_jump = self._run(psi_after_jump, _rand)
            # always do interpolation regardless of options.save_states
            # extract saved y values to interpolate
            psis_after_jump = result_after_jump.ys[0]
            # replace infs with nans to interpolate
            psi_inf_to_nan = jnp.where(
                jnp.isinf(psis_after_jump), jnp.nan, psis_after_jump
            )
            # setting fill_forward_nans_at_end prevents nan states from getting saved
            psi_coeffs = dx.backward_hermite_coefficients(
                new_times, psi_inf_to_nan, fill_forward_nans_at_end=True
            )
            psi_interpolator = dx.CubicInterpolation(new_times, psi_coeffs)

            final_time = result_after_jump.ts[1]
            final_state = result_after_jump.ys[1][0]

            def save_ts_and_states(_save_state):
                def inner_cond_fun(__save_state):
                    return (self.ts[__save_state.save_index] <= final_time) & (
                        __save_state.save_index < len(self.ts)
                    )

                def inner_body_fun(__save_state):
                    _t = self.ts[__save_state.save_index]
                    _y = psi_interpolator.evaluate(_t)
                    _ts = __save_state.ts.at[__save_state.save_index].set(_t)
                    _ys = jtu.tree_map(
                        lambda __y, __ys: __ys.at[__save_state.save_index].set(__y),
                        self.save(_y),
                        __save_state.ys,
                    )
                    return SaveState(
                        ts=_ts,
                        ys=_ys,
                        save_index=__save_state.save_index + 1,
                        jump_times=__save_state.jump_times,
                    )

                save_state = while_loop(
                    inner_cond_fun,
                    inner_body_fun,
                    _save_state,
                    max_steps=len(self.ts),
                    buffers=_inner_buffers,
                    kind='checkpointed',
                )
                return save_state

            save_state = save_ts_and_states(_state.save_state)
            jump_times = save_state.jump_times.at[_state.num_jumps].set(jump_time)
            save_state = SaveState(
                ts=save_state.ts,
                ys=save_state.ys,
                save_index=save_state.save_index,
                jump_times=jump_times,
            )
            return State(
                save_state=save_state,
                final_state=final_state,
                final_time=final_time,
                num_jumps=_state.num_jumps + 1,
                key=_next_key,
            )

        return while_loop(
            outer_cond_fun,
            outer_body_fun,
            state,
            max_steps=MAX_JUMPS,
            buffers=_outer_buffers,
            kind='checkpointed',
        )

    def sample_jump_ops(self, t, psi, key):
        """given a state psi at time t that should experience a jump,
        randomly sample one jump operator from among the provided jump_ops.
        The probability that a certain jump operator is selected is weighted
        by the probability that such a jump can occur. For instance for a qubit
        experiencing amplitude damping, if it is in the ground state then
        there is probability zero of experiencing an amplitude damping event.
        """
        Ls = jnp.stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        # i, j, k: hilbert dim indices; e: jump ops; d: index of dimension 1
        probs = jnp.einsum('id,eij,ejk,kd->e', jnp.conj(psi), Lsd, Ls, psi)
        # for categorical we pass in the log of the probabilities
        logits = jnp.log(jnp.real(probs / (jnp.sum(probs))))
        # randomly sample the index of a single jump operator
        sample_idx = jax.random.categorical(key, logits, shape=(1,))
        # extract that jump operator and squeeze size 1 dims
        return jnp.squeeze(jnp.take(Ls, sample_idx, axis=0), axis=0)


# fmt: off
# ruff: noqa
class MCSolveEulerIntegrator(MCSolveDiffraxIntegrator, EulerIntegrator): pass
class MCSolveDopri5Integrator(MCSolveDiffraxIntegrator, Dopri5Integrator): pass
class MCSolveDopri8Integrator(MCSolveDiffraxIntegrator, Dopri8Integrator): pass
class MCSolveTsit5Integrator(MCSolveDiffraxIntegrator, Tsit5Integrator): pass
class MCSolveKvaerno3Integrator(MCSolveDiffraxIntegrator, Kvaerno3Integrator): pass
class MCSolveKvaerno5Integrator(MCSolveDiffraxIntegrator, Kvaerno5Integrator): pass
# fmt: on
