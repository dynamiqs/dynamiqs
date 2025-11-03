from __future__ import annotations

from dataclasses import replace

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree

from ...options import Options
from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import JSSESolveResult, JumpSolveSaved, Result, Saved, SolveSaved
from ...utils.general import expect
from .abstract_integrator import StochasticBaseIntegrator
from .diffrax_integrator import call_diffeqsolve
from .interfaces import JSSEInterface, SolveInterface
from .save_mixin import SolveSaveMixin


class EventInfos(eqx.Module):
    noclick_states: QArray
    noclick_prob: float
    noclick_expects: Array | None

    def __str__(self) -> str:
        # return super().__str__()  # todo: custom print
        return 'EventInfos(...)'


class JumpState(eqx.Module):
    """State for the jump SSE event integrator."""

    psi: QArray  # state (integrated from initial to current time)
    t: float  # current time
    key: PRNGKeyArray  # active key
    clicktimes: Array  # click times of shape (nLs, nmaxclick)
    indices: Array  # last click time indices of shape (nLs,)
    saved: SolveSaved  # saved states, expectation values and extras
    save_index: int

    def new_click(self, idx: int, t: float) -> tuple[Array, Array]:
        clicktimes = self.clicktimes.at[idx, self.indices[idx]].set(t)
        indices = self.indices.at[idx].add(1)
        return clicktimes, indices

    def new_save(self, new: SolveSaved) -> JumpState:
        idx = self.save_index
        saved = jax.tree.map(
            lambda _new, _saved: _saved.at[idx].set(_new[idx]), new, self.saved
        )
        return replace(self, saved=saved, save_index=idx + 1)


class JSSESolveEventIntegrator(
    StochasticBaseIntegrator, JSSEInterface, SolveInterface, SolveSaveMixin
):
    """Integrator computing the time evolution of the Jump SSE using Diffrax events."""

    def run(self) -> Result:
        if self.method.smart_sampling:
            # sample a no-click trajectory and compute its probability
            solution = self._solve_noclick(self.ts, self.y0)
            psis = solution.ys[0]
            noclick_psis = psis.unit()
            noclick_prob = psis[-1].norm() ** 2
            noclick_expects = None if self.Es is None else expect(self.Es, noclick_psis)
            infos = EventInfos(noclick_psis, noclick_prob, noclick_expects)
        else:
            noclick_prob = None
            infos = None

        # vectorize over keys
        out_axes = JumpSolveSaved(0, 0, 0, 0)
        f = lambda key: self._solve_single_trajectory(key, noclick_prob)
        saved = jax.vmap(f, 0, out_axes)(self.key)
        return self.result(saved, infos)

    def _solve_noclick(
        self,
        ts: Array,
        psi0: Array,
        event: dx.Event | None = None,
        save: callable | None = None,
    ) -> dx.Solution:
        terms = dx.ODETerm(
            lambda t, y, _: -1j * self.H(t) @ y
            + sum([-0.5 * _L.dag() @ (_L @ y) for _L in self.L(t)])
        )

        return call_diffeqsolve(
            ts,
            psi0,
            terms,
            self.method.noclick_method,
            self.gradient,
            Options(progress_meter=False).initialise(),
            self.discontinuity_ts,
            event=event,
            save=save,
            dtmax=self.method.dtmax,
        )

    def save(self, y: PyTree) -> Saved:
        return super().save(y.unit())

    def _solve_until_click(
        self, y: JumpState, rand: float
    ) -> tuple[JumpState, SolveSaved, bool]:
        # === solve until the next click event
        cond_fn = lambda t, y, *a, **kw: y.norm() ** 2 - rand  # noqa: ARG005
        event = dx.Event(cond_fn, self.method.root_finder)
        ts = jnp.where(self.ts >= y.t, self.ts, y.t)  # clip times to start at y.t
        solution = self._solve_noclick(ts, y.psi, event=event, save=self.save)

        # === collect solve result
        new_saved = solution.ys[0]
        psiclick, tclick = solution.ys[1][0], solution.ts[1][0]
        y = replace(y, psi=psiclick, t=tclick)
        click_occurred = solution.event_mask

        # === save intermediate states, expectation values and extras
        save_cond = lambda y: (
            (self.ts[y.save_index] <= y.t) & (y.save_index < len(self.ts))
        )
        save_body = lambda y: y.new_save(new_saved)
        y = while_loop(
            save_cond,
            save_body,
            y,
            max_steps=len(self.ts),
            buffers=lambda x: x.saved,
            kind='checkpointed',
        )

        return y, click_occurred

    def _solve_single_trajectory(
        self, key: PRNGKeyArray, noclick_prob: float | None
    ) -> JSSESolveResult:
        def loop_body(y: JumpState) -> JumpState:
            # === pick a random number for the next click event
            newkey, key_click, key_jump_choice = jax.random.split(y.key, 3)
            y = replace(y, key=newkey)  # update key
            minval = 0.0
            if noclick_prob is not None:
                # when smart_sampling = True, and if no jump occurred yet we choose
                # minval = noclick_prob to ensure that all trajectories are sampled
                # with at least one click
                minval = jax.lax.cond(
                    jnp.any(y.indices > 0), lambda: 0.0, lambda: noclick_prob
                )
            rand = jax.random.uniform(key_click, minval=minval)

            # === solve until the next click event
            y, click_occurred = self._solve_until_click(y, rand)

            # === apply click
            # if the click triggered, update the quantum state and clicktimes,
            # else, the final time was reached, and do nothing
            def click(y: JumpState) -> JumpState:
                psi = y.psi
                L = stack(self.L(y.t))  # todo: remove stack
                LdL = L.dag() @ L
                exp_LdL = expect(LdL, psi).real
                L_probas = exp_LdL / sum(exp_LdL)

                # sample one of the jump operators
                idx = jax.random.choice(key_jump_choice, len(self.Ls), p=L_probas)

                # apply the jump operator
                psi = L[idx] @ psi / jnp.sqrt(exp_LdL[idx])

                # update click time and index
                clicktimes, indices = y.new_click(idx, y.t)

                return replace(y, psi=psi, clicktimes=clicktimes, indices=indices)

            skip = lambda y: y
            return jax.lax.cond(click_occurred, click, skip, y)

        # === prepare the initial state to loop over
        clicktimes = jnp.full((len(self.Ls), self.options.nmaxclick), jnp.nan)
        indices = jnp.zeros(len(self.Ls), dtype=int)
        y = stack([self.y0] * len(self.ts))
        saved = self.reorder_Esave(self.save(y))
        y0 = JumpState(self.y0, self.t0, key, clicktimes, indices, saved, 0)

        # === loop over no-click evolutions until the final time is reached
        loop_condition = lambda y: y.t < self.t1
        yend = while_loop(
            loop_condition,
            loop_body,
            y0,
            max_steps=self.options.nmaxclick,
            buffers=lambda x: x.saved,
            kind='checkpointed',
        )

        # === return result
        saved = self.postprocess_saved(yend.saved, yend.psi[None, ...])
        return JumpSolveSaved(saved.ysave, saved.extra, saved.Esave, yend.clicktimes)


jssesolve_event_integrator_constructor = JSSESolveEventIntegrator
