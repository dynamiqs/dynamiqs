from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ...gradient import Autograd, CheckpointAutograd
from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import Result
from ...utils.general import dag, expect, norm, unit
from .abstract_integrator import BaseIntegrator, StochasticBaseIntegrator
from .interfaces import (
    AbstractTimeInterface,
    JSSEInterface,
    MEInterface,
    SEInterface,
    SolveInterface,
)
from .save_mixin import (
    AbstractSaveMixin,
    JumpSolveSaveMixin,
    PropagatorSaveMixin,
    SolveSaveMixin,
)


class FixedStepInfos(eqx.Module):
    nsteps: Array

    def __str__(self) -> str:
        if self.nsteps.ndim >= 1:
            # note: fixed step methods always make the same number of steps
            return f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
        return f'{self.nsteps} steps'


class AdaptiveStepInfos(eqx.Module):
    nsteps: Array
    naccepted: Array
    nrejected: Array

    def __str__(self) -> str:
        if self.nsteps.ndim >= 1:
            return (
                f'avg. {self.nsteps.mean():.1f} steps ({self.naccepted.mean():.1f}'
                f' accepted, {self.nrejected.mean():.1f} rejected) | infos shape'
                f' {self.nsteps.shape}'
            )
        return (
            f'{self.nsteps} steps ({self.naccepted} accepted,'
            f' {self.nrejected} rejected)'
        )


class DiffraxIntegrator(BaseIntegrator, AbstractSaveMixin, AbstractTimeInterface):
    """Integrator using the Diffrax library."""

    diffrax_solver: dx.AbstractSolver
    fixed_step: bool

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        if self.fixed_step:
            return dx.ConstantStepSize()
        else:
            return dx.PIDController(
                rtol=self.method.rtol,
                atol=self.method.atol,
                safety=self.method.safety_factor,
                factormin=self.method.min_factor,
                factormax=self.method.max_factor,
                jump_ts=self.discontinuity_ts,
            )

    @property
    def dt0(self) -> float | None:
        return self.method.dt if self.fixed_step else None

    @property
    def max_steps(self) -> int:
        # TODO: fix hard-coded max_steps for fixed methods
        return 100_000 if self.fixed_step else self.method.max_steps

    @property
    @abstractmethod
    def terms(self) -> dx.AbstractTerm:
        pass

    @property
    def adjoint(self) -> dx.AbstractAdjoint:
        if self.gradient is None:
            return dx.RecursiveCheckpointAdjoint()
        elif isinstance(self.gradient, CheckpointAutograd):
            return dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
        elif isinstance(self.gradient, Autograd):
            return dx.DirectAdjoint()
        else:
            raise TypeError(f'Unknown gradient type {self.gradient}')

    def diffeqsolve(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        saveat: dx.SaveAt,
        event: dx.Event | None = None,
    ) -> dx.Solution:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)

            # === solve differential equation with diffrax
            return dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=t0,
                t1=t1,
                dt0=self.dt0,
                y0=y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=self.adjoint,
                event=event,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

    def run(self) -> Result:
        # === prepare diffrax arguments
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === solve differential equation
        solution = self.diffeqsolve(t0=self.t0, t1=self.t1, y0=self.y0, saveat=saveat)

        # === collect and return results
        saved = self.postprocess_saved(*solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

    def infos(self, stats: dict[str, Array]) -> PyTree:
        if self.fixed_step:
            return FixedStepInfos(stats['num_steps'])
        else:
            return AdaptiveStepInfos(
                stats['num_steps'],
                stats['num_accepted_steps'],
                stats['num_rejected_steps'],
            )


class SEDiffraxIntegrator(DiffraxIntegrator, SEInterface):
    """Integrator solving the Schrödinger equation with Diffrax."""

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class SEPropagatorDiffraxIntegrator(SEDiffraxIntegrator, PropagatorSaveMixin):
    """Integrator computing the propagator of the Schrödinger equation using the Diffrax
    library.
    """


sepropagator_euler_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
sepropagator_dopri5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
sepropagator_dopri8_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
sepropagator_tsit5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
sepropagator_kvaerno3_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
sepropagator_kvaerno5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)


class SESolveDiffraxIntegrator(SEDiffraxIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Schrödinger equation using the
    Diffrax library.
    """


sesolve_euler_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
sesolve_dopri5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
sesolve_dopri8_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
sesolve_tsit5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
sesolve_kvaerno3_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
sesolve_kvaerno5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)


class MEDiffraxIntegrator(DiffraxIntegrator, MEInterface):
    """Integrator solving the Lindblad master equation with Diffrax."""

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation for a single loss channel is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            L, H = self.L(t), self.H(t)
            Hnh = -1j * H + sum([-0.5 * _L.dag() @ _L for _L in L])
            tmp = Hnh @ y + sum([0.5 * _L @ y @ _L.dag() for _L in L])
            return tmp + tmp.dag()

        return dx.ODETerm(vector_field)


class MESolveDiffraxIntegrator(MEDiffraxIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Diffrax library.
    """


mesolve_euler_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
mesolve_dopri5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
mesolve_dopri8_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
mesolve_tsit5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
mesolve_kvaerno3_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
mesolve_kvaerno5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)


class JSSEInnerState(eqx.Module):
    saved: PyTree
    save_index: int


class JSSEState(eqx.Module):
    y: QArray  # quantum state
    t: Scalar  # time
    key: PRNGKeyArray  # active key
    numclicks: int  # number of clicks
    clicktimes: Array  # time of clicks
    inner_state: JSSEInnerState  # saved quantities


def save_buffers(inner_state: JSSEInnerState) -> PyTree:
    assert type(inner_state) is JSSEInnerState
    return inner_state.saved


def loop_buffers(state: JSSEState) -> PyTree:
    assert type(state) is JSSEState
    return state.inner_state.saved


class JSSESolveEventIntegrator(
    StochasticBaseIntegrator,
    DiffraxIntegrator,
    JSSEInterface,
    SolveInterface,
    JumpSolveSaveMixin,
):
    """Integrator computing the time evolution of the Jump SSE using Diffrax events."""

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            L, H = self.L(t), self.H(t)
            return -1j * H @ y + sum([-0.5 * _L.dag() @ (_L @ y) for _L in L])

        return dx.ODETerm(vector_field)

    def run(self) -> Result:
        def loop_condition(state: JSSEState) -> bool:
            return state.t < self.t1

        def loop_body(state: JSSEState) -> JSSEState:
            # pick a random number for the next detection event
            key, click_key, jump_key = jax.random.split(state.key, num=3)
            rand = jax.random.uniform(click_key)

            # solve until the next detection event
            solution = self._solve_until_click(state.y, state.t, rand)
            temp_saved = solution.ys[0]
            yclick = solution.ys[1][0]
            tclick = solution.ts[1][0]

            # if the event triggered, update the quantum state and clicktimes
            # else, the final time was reached, and do nothing
            def event(
                yclick: QArray, clicktimes: Array, numclicks: int
            ) -> tuple[QArray, Array, int]:
                # find a random jump operator among the provided jump_ops, and apply it
                jump_op, idx = self._sample_jump_ops(tclick, yclick, jump_key)
                yclick = unit(jump_op @ yclick)

                # update clicktimes
                clicktimes = clicktimes.at[idx, state.numclicks].set(tclick)
                numclicks += 1

                return yclick, clicktimes, numclicks

            def skip(
                yclick: QArray, clicktimes: Array, numclicks: int
            ) -> tuple[QArray, Array, int]:
                return yclick, clicktimes, numclicks

            yclick, clicktimes, numclicks = jax.lax.cond(
                solution.event_mask,
                event,
                skip,
                yclick,
                state.clicktimes,
                state.numclicks,
            )

            # save intermediate states and expectation values, based on the code
            # in diffrax/_integrate.py which can be found at:
            # https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py#L427-L458
            def save_cond(inner_state: JSSEInnerState) -> bool:
                save_index = inner_state.save_index
                return (self.ts[save_index] <= tclick) & (save_index < len(self.ts))

            def save_body(inner_state: JSSEInnerState) -> JSSEInnerState:
                idx = inner_state.save_index
                saved = jax.tree.map(
                    lambda _temp_saved, _saved: _saved.at[idx].set(_temp_saved[idx]),
                    temp_saved,
                    inner_state.saved,
                )
                return JSSEInnerState(saved, idx + 1)

            inner_state = while_loop(
                save_cond,
                save_body,
                state.inner_state,
                max_steps=len(self.ts),
                buffers=save_buffers,
                kind='checkpointed',
            )

            # return updated state
            return JSSEState(yclick, tclick, key, numclicks, clicktimes, inner_state)

        # prepare the initial state to loop over
        y = stack([self.y0] * len(self.ts))
        saved = self.reorder_Esave(self.save(y))
        clicktimes = jnp.full((len(self.Ls), self.options.nmaxclick), jnp.nan)
        inner_state = JSSEInnerState(saved, 0)
        state = JSSEState(self.y0, self.t0, self.key, 0, clicktimes, inner_state)

        # loop over no jump evolutions until the final time is reached
        final_state = while_loop(
            loop_condition,
            loop_body,
            state,
            kind='checkpointed',
            buffers=loop_buffers,
            max_steps=self.options.nmaxclick,
        )

        # collect and return results
        saved = final_state.inner_state.saved  # of type SolveSaved
        clicktimes = final_state.clicktimes
        saved = self.postprocess_saved(saved, clicktimes)  # of type JumpSolveSaved
        return self.result(saved, infos=None)

    def _solve_until_click(self, y0: QArray, t0: Array, rand: Array) -> dx.Solution:
        # === prepare saveat
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        ts = jnp.where(self.ts >= t0, self.ts, t0)  # clip ts to start at t0
        subsaveat_a = dx.SubSaveAt(ts=ts, fn=fn)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === prepare event
        def cond_fn(t: Scalar, y: QArray, *args, **kwargs) -> Array:
            return norm(y) ** 2 - rand

        event = dx.Event(cond_fn, self.options.root_finder)

        # === solve differential equation with diffrax
        return self.diffeqsolve(t0=t0, t1=self.t1, y0=y0, saveat=saveat, event=event)

    def _sample_jump_ops(self, t: Array, psi: Array, key: Array) -> tuple[Array, int]:
        # given a state psi at time t that should experience a jump,
        # randomly sample one jump operator from among the provided jump_ops.
        # The probability that a certain jump operator is selected is weighted
        # by the probability that such a jump can occur. For instance for a qubit
        # experiencing amplitude damping, if it is in the ground state then
        # there is probability zero of experiencing an amplitude damping event.

        Ls = stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        probs = expect(Lsd @ Ls, psi)
        # for categorical we pass in the log of the probabilities
        logits = jnp.log(jnp.real(probs / (jnp.sum(probs))))
        # randomly sample the index of a single jump operator
        sample_idx = jax.random.categorical(key, logits, shape=(1,))[0]
        return Ls[sample_idx], sample_idx


jssesolve_event_euler_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
jssesolve_event_dopri5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
jssesolve_event_dopri8_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
jssesolve_event_tsit5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
jssesolve_event_kvaerno3_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
jssesolve_event_kvaerno5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)
