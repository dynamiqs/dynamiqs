from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import equinox as eqx

from ...utils.utils import dag
from ..core.abstract_integrator import MCSolveIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)

class SaveState(eqx.Module):
    saveat_ts_index: int
    ts: Array
    ys: PyTree
    save_index: int


class MCSolveDiffraxIntegrator(DiffraxIntegrator, MCSolveIntegrator):

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t: Scalar, state: PyTree, _args: PyTree) -> PyTree:
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(axis=0)
            new_state = -1j * (self.H(t) - 1j * 0.5 * LdL) @ state
            return new_state
        return dx.ODETerm(vector_field)

    def event(self):
        def norm_below_rand(t, y, *args, **kwargs):
            prob = jnp.abs(jnp.einsum("id,id->", jnp.conj(y), y))
            return prob - self.rand
        return dx.Event(norm_below_rand, self.root_finder)

    def run(self) -> PyTree:
        """loop over jumps until the simulation reaches the final time"""

        def outer_while_cond(_save_state):
            return prev_result.final_time < tsave[-1]

        def outer_while_body(t_state_key_solver):
            prev_result, prev_t_jump, prev_num_jumps, prev_key = t_state_key_solver
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
            # TODO inner while loop will go here which saves intermediate results
            t_jump = jnp.where(
                next_result.final_time < tsave[-1], next_result.final_time, prev_t_jump
            )
            return next_result, t_jump, prev_num_jumps + 1, loop_key

        # solve until the first jump occurs. Enter the while loop for additional jumps
        key_1, key_2 = jax.random.split(key)
        initial_result = _jump_trajs(
            H, jump_ops, psi0, tsave, key_1, rand, exp_ops, solver, root_finder, gradient, options
        )

        final_result, last_t_jump, total_num_jumps, _ = while_loop(
            while_cond,
            while_body,
            (initial_result, initial_result.final_time, 0, key_2),
            max_steps=10,
            kind="bounded",
        )
        return final_result, last_t_jump, total_num_jumps


class MCSolveEulerIntegrator(MCSolveDiffraxIntegrator, EulerIntegrator):
    pass


class MCSolveDopri5Integrator(MCSolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MCSolveDopri8Integrator(MCSolveDiffraxIntegrator, Dopri8Integrator):
    pass


class MCSolveTsit5Integrator(MCSolveDiffraxIntegrator, Tsit5Integrator):
    pass


class MCSolveKvaerno3Integrator(MCSolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MCSolveKvaerno5Integrator(MCSolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
