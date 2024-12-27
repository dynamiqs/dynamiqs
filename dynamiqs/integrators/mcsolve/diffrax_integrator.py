from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
import jax
import equinox as eqx
import jax.tree_util as jtu
from equinox.internal import while_loop

from jax import Array
from jaxtyping import PyTree

from ..core.save_mixin import SolveSaveMixin
from ...utils import dag, unit, norm, expect
from ...qarrays import QArray
from ...qarrays.utils import stack
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
    key: Array


def _inner_buffers(save_state):
    assert type(save_state) is SaveState
    return save_state.ts, save_state.ys, save_state.jump_times


def _outer_buffers(state):
    assert type(state) is State
    save_state = state.save_state
    return save_state.ts, save_state.ys, save_state.jump_times


class MCSolveDiffraxIntegrator(MCDiffraxIntegrator, MCSolveIntegrator, SolveSaveMixin):
    """Integrator computing the time evolution of the Monte-Carlo unraveling of the
    Lindblad master equation using the Diffrax library."""

    def _solve_until_jump(self, y0: QArray, t0: Array, tsave: Array, rand: Array):
        # === prepare saveat
        subsaveat_a = dx.SubSaveAt(ts=tsave)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === prepare event
        def cond_fn(t, y, *args, **kwargs):
            return norm(y) ** 2 - rand

        event = dx.Event(cond_fn, self.root_finder)

        # === solve differential equation with diffrax
        return self.diffeqsolve(t0=t0, t1=self.t1, y0=y0, saveat=saveat, event=event)

    def run(self):
        # === run no jump, extract no-jump probability
        no_jump_solution = self._solve_until_jump(self.y0, self.t0, self.ts, 0.0)
        no_jump_saved = jax.vmap(self.save)(unit(no_jump_solution.ys[0]))
        no_jump_prob = norm(no_jump_solution.ys[1][0]) ** 2

        # === run jump trajectories
        jump_state_fun = jax.vmap(self._loop_over_jumps, in_axes=(0, None))
        jump_state = jump_state_fun(self.keys, no_jump_prob)
        jump_times = jump_state.save_state.jump_times
        num_jumps = jump_state.num_jumps

        # === save and postprocess results
        no_jump_saved = self.postprocess_saved(
            no_jump_saved, unit(no_jump_solution.ys[1])
        )
        no_jump_result = self.traj_result(
            no_jump_saved, infos=self.infos(no_jump_solution.stats)
        )
        jump_saved = self.postprocess_saved(
            jump_state.save_state.ys, jump_state.final_state[None]
        )
        # TODO save stats for jumps?
        jump_result = self.traj_result(jump_saved)

        return self.result(
            no_jump_result, jump_result, no_jump_prob, jump_times, num_jumps
        )

    def _loop_over_jumps(self, key: Array, no_jump_prob: Array) -> State:
        """loop over jumps until the simulation reaches the final time"""
        rand_key, sample_key = jax.random.split(key)
        rand = jax.random.uniform(rand_key, minval=no_jump_prob)
        first_result = self._solve_until_jump(self.y0, self.t0, self.ts, rand)
        first_jump_time = first_result.ts[1][0]
        time_diffs = self.ts - first_jump_time
        # want to start with a time after the jump
        time_diffs = jnp.where(time_diffs < 0, jnp.inf, time_diffs)
        save_index = jnp.argmin(time_diffs)
        # allocate memory for the state that is updated during the loop
        save_state = SaveState(
            ts=first_result.ts[0],
            ys=jax.vmap(self.save)(unit(first_result.ys[0])),
            save_index=save_index,
            jump_times=jnp.full(self.options.max_jumps, jnp.nan),
        )
        state = State(
            save_state=save_state,
            final_state=first_result.ys[1][0, :, :],
            final_time=first_jump_time,
            num_jumps=0,
            key=sample_key,
        )

        def _outer_cond_fun(_state):
            return _state.final_time < self.t1

        def _outer_body_fun(_state):
            jump_time = _state.final_time
            psi = _state.final_state
            _next_key, _rand_key, _sample_key = jax.random.split(_state.key, num=3)
            jump_op = self._sample_jump_ops(jump_time, psi, _sample_key)
            psi_after_jump = unit(jump_op @ psi)
            # Construct a new linspace over the remaining times with the same shape as
            # self.ts (because the shape can't be dynamic!). Note that jax does not
            # support dynamic array sizes: this makes vmaps of loops tricky as discussed
            # here https://github.com/patrick-kidger/equinox/issues/870. The issue is
            # that some trajectories terminate before others do (they experience fewer
            # jumps), but the loop body needs to keep executing until the last
            # trajectory finishes. In these "idling" cases, new_times is a constant
            # array with fill value self.t1. The interpolator
            # dx.backward_hermite_coefficients below requires the time inputs to be
            # monotonically strictly increasing, which would then throw an error. So in
            # these cases we pass a placeholder of self.ts.
            new_times = jnp.linspace(jump_time, self.t1, len(self.ts))
            new_times = jnp.where(_state.final_time < self.t1, new_times, self.ts)
            # This new random value should be uniform over [0, 1), not restricted to
            # [no_jump_prob, 1) like in the first case.
            _rand = jax.random.uniform(_rand_key)
            result_after_jump = self._solve_until_jump(
                psi_after_jump, new_times[0], new_times, _rand
            )
            # extract saved y values to interpolate, replace infs with nans
            psis_after_jump = result_after_jump.ys[0]
            psi_inf_to_nan = jtu.tree_map(
                lambda _y: jnp.where(jnp.isinf(_y), jnp.nan, _y), psis_after_jump
            )
            # setting fill_forward_nans_at_end prevents nan states from getting saved
            psi_coeffs = dx.backward_hermite_coefficients(
                new_times, psi_inf_to_nan, fill_forward_nans_at_end=True
            )
            psi_interpolator = dx.CubicInterpolation(new_times, psi_coeffs)

            final_time = result_after_jump.ts[1][0]
            final_state = result_after_jump.ys[1][0]

            def save_ts_and_states(_save_state: SaveState):
                # Save intermediate states and expectation values. Based on the code
                # in diffrax/_integrate.py which can be found here https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py#L427-L458  # noqa E501
                def _inner_cond_fun(__save_state):
                    return (self.ts[__save_state.save_index] <= final_time) & (
                        __save_state.save_index < len(self.ts)
                    )

                def _inner_body_fun(__save_state):
                    _t = self.ts[__save_state.save_index]
                    _y = psi_interpolator.evaluate(_t)
                    _ts = __save_state.ts.at[__save_state.save_index].set(_t)
                    _ys = jtu.tree_map(
                        lambda __y, __ys: __ys.at[__save_state.save_index].set(__y),
                        self.save(unit(_y)),
                        __save_state.ys,
                    )
                    return SaveState(
                        ts=_ts,
                        ys=_ys,
                        save_index=__save_state.save_index + 1,
                        jump_times=__save_state.jump_times,
                    )

                save_state = while_loop(
                    _inner_cond_fun,
                    _inner_body_fun,
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
            _outer_cond_fun,
            _outer_body_fun,
            state,
            max_steps=self.options.max_jumps,
            buffers=_outer_buffers,
            kind='checkpointed',
        )

    def _sample_jump_ops(self, t: Array, psi: Array, key: Array) -> Array:
        """given a state psi at time t that should experience a jump,
        randomly sample one jump operator from among the provided jump_ops.
        The probability that a certain jump operator is selected is weighted
        by the probability that such a jump can occur. For instance for a qubit
        experiencing amplitude damping, if it is in the ground state then
        there is probability zero of experiencing an amplitude damping event.
        """
        Ls = stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        probs = expect(Lsd @ Ls, psi)
        # for categorical we pass in the log of the probabilities
        logits = jnp.log(jnp.real(probs / (jnp.sum(probs))))
        # randomly sample the index of a single jump operator
        sample_idx = jax.random.categorical(key, logits, shape=(1,))[0]
        return Ls[sample_idx]


# fmt: off
# ruff: noqa
class MCSolveEulerIntegrator(MCSolveDiffraxIntegrator, EulerIntegrator): pass
class MCSolveDopri5Integrator(MCSolveDiffraxIntegrator, Dopri5Integrator): pass
class MCSolveDopri8Integrator(MCSolveDiffraxIntegrator, Dopri8Integrator): pass
class MCSolveTsit5Integrator(MCSolveDiffraxIntegrator, Tsit5Integrator): pass
class MCSolveKvaerno3Integrator(MCSolveDiffraxIntegrator, Kvaerno3Integrator): pass
class MCSolveKvaerno5Integrator(MCSolveDiffraxIntegrator, Kvaerno5Integrator): pass
# fmt: on
