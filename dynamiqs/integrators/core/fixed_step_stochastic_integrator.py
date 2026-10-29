from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar

from ...gradient import BackwardCheckpointed, Direct, Forward
from ...method import Rouchon1, _DEFixedStep
from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import DiffusiveSolveSaved, JumpSolveSaved, Result, SolveSaved
from ...utils.general import dag, expect
from ...utils.operators import eye_like
from .abstract_integrator import StochasticBaseIntegrator
from .interfaces import (
    AbstractTimeInterface,
    DSMEInterface,
    DSSEInterface,
    JSMEInterface,
    JSSEInterface,
    SolveInterface,
)
from .rouchon_integrator import RouchonPropertiesMixin, cholesky_normalize
from .save_mixin import SolveSaveMixin

# Equinox needs a number of checkpoints for a loop whose number of steps is unknown at
# trace time. This is its default for 100_000 steps, the bound of the fixed step ODE
# methods.
_DEFAULT_NCHECKPOINTS = 447
_DIRECT_MAX_STEPS = 100_000


class SDEState(eqx.Module):
    """State for the jump/diffusive SSE/SME fixed step integrators."""

    state: QArray


class StochasticSolveFixedStepIntegrator(
    StochasticBaseIntegrator, SolveInterface, SolveSaveMixin, AbstractTimeInterface
):
    """Integrator solving the jump/diffusive SSE/SME with a fixed step size
    integrator.
    """

    def __check_init__(self):
        # check that options.t0 is not used
        if self.options.t0 is not None:
            raise ValueError(
                'Option `t0` is invalid for fixed step SSE or SME methods.'
            )

        if len(self.discontinuity_ts) > 0:
            warnings.warn(
                'The Hamiltonian or jump operators are time-dependent with '
                'discontinuities, which will be ignored by the method.',
                stacklevel=1,
            )

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step methods always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    @property
    def dt(self) -> float:
        method = cast(_DEFixedStep, self.method)
        return method.dt

    @abstractmethod
    def sample_rv(self, key: PRNGKeyArray) -> PyTree:
        # sample the random variable driving a single step
        pass

    @abstractmethod
    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        # return SDE state y_{t+dt} from a random variable sample dX and the current
        # state y_t
        pass

    @abstractmethod
    def sde_y0(self) -> SDEState:
        # define initial SDE state
        pass

    def integrate(self, y: SDEState, n: Array, nend: Array) -> tuple[SDEState, Array]:
        # integrate the SDE from step index n to nend

        def body(carry):  # noqa: ANN001, ANN202
            y, n = carry
            # the noise of a step only depends on its index, so that the trajectory
            # does not depend on `tsave`
            dX = self.sample_rv(jax.random.fold_in(self.key, n))
            return self.forward(self.t0 + n * self.dt, y, dX), n + 1

        # The number of steps depends on `tsave`, which is traced, so the loop kind
        # sets which autodiff mode is supported.
        max_steps, checkpoints = None, None
        if isinstance(self.gradient, Forward):
            kind = 'lax'
        elif isinstance(self.gradient, Direct):
            kind, max_steps = 'bounded', _DIRECT_MAX_STEPS
        else:
            ncheckpoints = _DEFAULT_NCHECKPOINTS
            if isinstance(self.gradient, BackwardCheckpointed):
                ncheckpoints = self.gradient.ncheckpoints or ncheckpoints
            # each save interval keeps its own checkpoints for the backward pass, so
            # the budget is split between them
            nintervals = max(len(self.ts) - 1, 1)
            kind, checkpoints = 'checkpointed', max(ncheckpoints // nintervals, 1)

        cond = lambda carry: carry[1] < nend
        return eqx.internal.while_loop(
            cond, body, (y, n), kind=kind, max_steps=max_steps, checkpoints=checkpoints
        )

    def run(self) -> Result:
        # index n of the step ending at each save time t = t0 + n dt, `tsave` values
        # are exact multiples of dt (see `check_times_multiple_of()`)
        save_steps = jnp.round((self.ts - self.t0) / self.dt).astype(int)
        if isinstance(self.gradient, Direct):
            save_steps = eqx.error_if(
                save_steps,
                save_steps[-1] > _DIRECT_MAX_STEPS,
                f'Gradient `Direct` supports at most {_DIRECT_MAX_STEPS:_} steps of'
                ' fixed step SSE or SME methods. Use `dq.gradient.BackwardCheckpointed`'
                ' or `dq.gradient.Forward` instead.',
            )

        # === initial state
        y0 = self.sde_y0()
        saved0 = self.save(self.ts[0], y0)

        # === integrate the SSE/SME for each save interval
        def outer_step(carry, x):  # noqa: ANN001, ANN202
            t, nend = x
            y, n = self.integrate(*carry, nend)
            return (y, n), self.save(t, y)

        xs = (self.ts[1:], save_steps[1:])
        (ylast, nsteps), saved = jax.lax.scan(outer_step, (y0, jnp.asarray(0)), xs)

        # === collect and format results
        # insert the initial saved result
        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        saved = self.postprocess_saved(saved, ylast)
        return self.result(saved, infos=self.Infos(nsteps))


class JumpState(SDEState):
    """State for the jump SSE/SME fixed step integrators."""

    state: QArray  # state (integrated from initial to current time)
    clicktimes: Array  # click times of shape (nmeas, nmaxclick)
    nclicks: Array  # number of clicks of shape (nmeas,)

    def record_click(
        self, state: QArray, dN: Array, idx: Array, t: RealScalarLike
    ) -> JumpState:
        # record a click of the idx-th detector at time t if dN = 1, no click (dN = 0)
        # or a click beyond `nmaxclick` writes out of bounds and is dropped
        nclick = jnp.where(dN == 1, self.nclicks[idx], self.clicktimes.shape[1])
        clicktimes = self.clicktimes.at[idx, nclick].set(t, mode='drop')
        return JumpState(state, clicktimes, self.nclicks.at[idx].add(dN))


class JumpSolveFixedStepIntegrator(StochasticSolveFixedStepIntegrator):
    """Integrator solving the jump SSE/SME with a fixed step size integrator."""

    @property
    @abstractmethod
    def nmeas(self) -> int:
        pass

    def sde_y0(self) -> SDEState:
        clicktimes = jnp.full((self.nmeas, self.options.nmaxclick), jnp.nan)
        return JumpState(self.y0, clicktimes, jnp.zeros(self.nmeas, dtype=int))

    def sample_rv(self, key: PRNGKeyArray) -> PyTree:
        # Sample two uniforms between 0 and 1. The first value is used to sample a
        # click or no click, the second value is used to sample one of the jump
        # operator when a click occurs.
        return jax.random.uniform(key, (2,))

    def save(self, t: Scalar, y: PyTree) -> SolveSaved:
        return super().save(t, y.state)

    def postprocess_saved(self, saved: SolveSaved, ylast: PyTree) -> JumpSolveSaved:
        saved = super().postprocess_saved(saved, ylast.state[None, :])
        return JumpSolveSaved(saved.ysave, saved.extra, saved.Esave, ylast.clicktimes)


class JSSESolveFixedStepIntegrator(JumpSolveFixedStepIntegrator, JSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


def _nonzero(x: ArrayLike) -> Array:
    # Replace zeros by ones, to divide by x when the click probability is zero (the
    # quotient is then zero or discarded). Guarding the quotient with `jnp.where`
    # instead gives NaN reverse-mode gradients (0 * inf).
    return jnp.where(x == 0.0, 1.0, x)


class JSSESolveEulerJumpIntegrator(JSSESolveFixedStepIntegrator):
    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        L, H = self.L(t), self.H(t)
        L = stack(L)  # todo: remove stack
        LdL = L.dag() @ L

        exp_LdL = expect(LdL, psi).real

        # === click probabilities
        tmp = exp_LdL * self.dt
        any_click_proba = sum(tmp)
        # avoid division by zero when the click probability is zero
        click_probas = tmp / _nonzero(any_click_proba)

        # === click or no click?
        dN = jax.lax.select(dX[0] < any_click_proba, 1, 0)

        # === click
        # sample one of the jump operators
        cum_probas = jnp.cumsum(click_probas)
        jump_idx = jnp.searchsorted(
            cum_probas, dX[1], side='right', method='compare_all'
        )
        # avoid division by zero when the click probability is zero
        normalisation = 1 / jnp.sqrt(_nonzero(exp_LdL[jump_idx]))
        # apply the jump operator
        psi_click = L[jump_idx] @ psi * normalisation

        # === no click
        # An Euler step of the nonlinear drift -(iH + 0.5 (LdL - <LdL>)) is unstable at
        # ||psi|| = 1, d||psi||^2 = <LdL> (||psi||^2 - 1) dt, so we apply the linear
        # no-click operator and renormalise instead.
        psi_noclick = (psi - self.dt * (1j * H + 0.5 * sum(LdL)) @ psi).unit()

        # === update state and click record
        psi = dN * psi_click + (1 - dN) * psi_noclick
        return cast(JumpState, y).record_click(psi, dN, jump_idx, t)


jssesolve_euler_jump_integrator_constructor = JSSESolveEulerJumpIntegrator


class JSMESolveFixedStepIntegrator(JumpSolveFixedStepIntegrator, JSMEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.etas)


class JSMESolveEulerJumpIntegrator(JSMESolveFixedStepIntegrator):
    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        # The jump SME for a single detector is:
        #   drho = Lcal(rho) dt
        #        + (Ccal(rho) / Tr[Ccal(rho)] - rho) (dN - Tr[Ccal(rho)] dt)
        #   P[dN=1] = Tr[Ccal(rho)] dt
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = theta rho + eta L @ rho @ Ld

        rho = y.state
        L, H = self.L(t), self.H(t)
        L = stack(L)  # todo: remove stack

        # === Ccal(rho)
        L_rho_Ldag = L @ rho @ L.dag()  # (nL, n, n)
        # the monitored operators are the last nLm entries of L = Lc + Lm
        Lm_rho_Lmdag = L_rho_Ldag[len(self.Lcs) :]  # (nLm, n, n)
        thetas = self.thetas[:, None, None]  # (nLm, 1, 1)
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = thetas * rho + etas * Lm_rho_Lmdag  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        # === click probabilities
        tmp = tr_Ccal_rho * self.dt
        any_click_proba = sum(tmp)
        # avoid division by zero when the click probability is zero
        click_probas = tmp / _nonzero(any_click_proba)

        # === click or no click?
        dN = jax.lax.select(dX[0] < any_click_proba, 1, 0)

        # === click
        # sample one of the jump operators
        cum_probas = jnp.cumsum(click_probas)
        jump_idx = jnp.searchsorted(
            cum_probas, dX[1], side='right', method='compare_all'
        )
        # avoid division by zero when the click probability is zero
        normalisation = 1 / _nonzero(tr_Ccal_rho[jump_idx])
        # apply the measurement backaction superoperator
        rho_click = Ccal_rho[jump_idx] * normalisation

        # === no click
        # compute Lcal(rho), see `MESolveDiffraxIntegrator` in
        # `integrators/core/diffrax_integrator.py`
        Hnh = -1j * H - 0.5 * sum(L.dag() @ L)
        tmp = Hnh @ rho + 0.5 * sum(L_rho_Ldag)
        Lcal_rho = tmp + dag(tmp)
        rho_noclick = (
            rho
            + Lcal_rho * self.dt
            + self.dt * sum(tr_Ccal_rho[:, None, None] * rho - Ccal_rho)
        )

        # === update state and click record
        rho = dN * rho_click + (1 - dN) * rho_noclick
        return cast(JumpState, y).record_click(rho, dN, jump_idx, t)


jsmesolve_euler_jump_integrator_constructor = JSMESolveEulerJumpIntegrator


class DiffusiveState(SDEState):
    """State for the diffusive SSE/SME fixed step integrators."""

    state: QArray  # state (integrated from initial to current time)
    Y: Array  # measurement (integrated from initial to current time)


class DiffusiveSolveFixedStepIntegrator(StochasticSolveFixedStepIntegrator):
    """Integrator solving the diffusive SSE/SME with a fixed step size integrator."""

    @property
    @abstractmethod
    def nmeas(self) -> int:
        pass

    def sde_y0(self) -> SDEState:
        # define initial SDE state (state, Y) = (state0, 0)
        return DiffusiveState(self.y0, jnp.zeros(self.nmeas))

    def sample_rv(self, key: PRNGKeyArray) -> PyTree:
        return jnp.sqrt(self.dt) * jax.random.normal(key, (self.nmeas,))

    def save(self, t: Scalar, y: PyTree) -> DiffusiveSolveSaved:
        saved = super().save(t, y.state)
        return DiffusiveSolveSaved(saved.ysave, saved.extra, saved.Esave, y.Y)

    def postprocess_saved(
        self, saved: SolveSaved, ylast: PyTree
    ) -> DiffusiveSolveSaved:
        saved = cast(
            DiffusiveSolveSaved, super().postprocess_saved(saved, ylast.state[None, :])
        )

        # The averaged measurement I^{(ta, tb)} is recovered by diffing the measurement
        # I which is integrated between ts[0] and ts[-1]
        Isave = jnp.diff(saved.Isave, axis=0) / jnp.diff(self.ts)[:, None]

        # reorder Isave after jax.lax.scan stacking (ntsave, nLm) -> (nLm, ntsave)
        Isave = Isave.swapaxes(-1, -2)
        return eqx.tree_at(lambda x: x.Isave, saved, Isave)


class DSSEFixedStepIntegrator(DiffusiveSolveFixedStepIntegrator, DSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


class DSSESolveEulerMayuramaIntegrator(DSSEFixedStepIntegrator):
    """Integrator solving the diffusive SSE with the Euler-Mayurama method."""

    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        dW = dX
        L, H = self.L(t), self.H(t)
        Lpsi = [_L @ psi for _L in L]

        # === measurement Y
        # dY = <L+Ld> dt + dW
        # small trick to compute <L+Ld>
        #   <L+Ld> = <psi|L+Ld|psi >
        #          = psi.dag() @ (L + Ld) @ psi
        #          = psi.dag() @ L @ psi + (psi.dag() @ L @ psi).dag()
        #          = 2 Re[psi.dag() @ Lpsi]
        exp = 2 * expect(L, psi).real  # (nL)
        dY = exp * self.dt + dW

        # === state psi
        dpsi = (
            -1j * self.dt * H @ psi
            - 0.5
            * self.dt
            * sum(
                [
                    _L.dag() @ _Lpsi - _exp * _Lpsi + 0.25 * _exp**2 * psi
                    for _L, _Lpsi, _exp in zip(L, Lpsi, exp, strict=True)
                ]
            )
            + sum(
                [
                    (_Lpsi - 0.5 * _exp * psi) * _dW
                    for _Lpsi, _exp, _dW in zip(Lpsi, exp, dW, strict=True)
                ]
            )
        )

        y = cast(DiffusiveState, y)
        return DiffusiveState(psi + dpsi, y.Y + dY)


def cholesky_normalize_ket(S: QArray, psi: QArray) -> jax.Array:
    # See comment of `cholesky_normalize()`.
    # For a ket we compute ~M @ psi = M @ T^{†(-1)} @ psi, so we directly replace psi by
    # T^{†(-1)} @ psi.

    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    _psi = psi.to_jax()[:, 0]  # (n, 1) -> (n,)
    # solve T^† @ x = psi => x = T^{†(-1)} @ psi
    return jax.lax.linalg.triangular_solve(
        T, _psi, lower=True, transpose_a=True, conjugate_a=True, left_side=True
    )[:, None]  # (n,) -> (n, 1)


class DSSESolveRouchon1Integrator(RouchonPropertiesMixin, DSSEFixedStepIntegrator):
    """Integrator solving the diffusive SSE with the Rouchon1 method."""

    method: Rouchon1

    @property
    def identity(self) -> QArray:
        return eye_like(self.H(0))

    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        dW = dX
        L, H = self.L(t + self.dt / 2), self.H(t + self.dt / 2)

        # === measurement Y
        # dY = <L+Ld> dt + dW
        # small trick to compute <L+Ld>
        #   <L+Ld> = <psi|L+Ld|psi >
        #          = psi.dag() @ (L + Ld) @ psi
        #          = psi.dag() @ L @ psi + (psi.dag() @ L @ psi).dag()
        #          = 2 Re[psi.dag() @ Lpsi]
        exp = 2 * expect(L, psi).real  # (nL)
        dY = exp * self.dt + dW

        if self.method.normalize:
            M0 = (
                self.identity
                - (1j * H + 0.5 * sum(_L.dag() @ _L for _L in L)) * self.dt
            )
            M_dY = M0 + sum([_dY * _L for _dY, _L in zip(dY, L, strict=True)])
            S = M0.dag() @ M0 + sum([_L.dag() @ _L for _L in L]) * self.dt

            psi = cholesky_normalize_ket(S, psi)
            psi = (M_dY @ psi).unit()

        else:
            # avoid computing the operators Ms_average and S,
            # which can be costly for large systems
            Lpsi = [(_L @ psi) for _L in L]
            M0psi = (
                psi
                + (
                    -1j * H @ psi
                    - 0.5
                    * sum(
                        [_L.dag() @ (_Lpsi) for _L, _Lpsi in zip(L, Lpsi, strict=True)]
                    )
                )
                * self.dt
            )
            psi = M0psi + sum(
                [_Lpsi * _dY for _Lpsi, _dY in zip(Lpsi, dY, strict=True)]
            )
            psi = psi.unit()  # normalise by norm

        y = cast(DiffusiveState, y)
        return DiffusiveState(psi, y.Y + dY)


dssesolve_euler_maruyama_integrator_constructor = DSSESolveEulerMayuramaIntegrator
dssesolve_rouchon1_integrator_constructor = DSSESolveRouchon1Integrator


class DSMEFixedStepIntegrator(DiffusiveSolveFixedStepIntegrator, DSMEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.etas)


class DSMESolveEulerMayuramaIntegrator(DSMEFixedStepIntegrator):
    """Integrator solving the diffusive SME with the Euler-Mayurama method."""

    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        # The diffusive SME for a single detector is:
        #   drho = Lcal(rho)     dt + (Ccal(rho) - Tr[Ccal(rho)] rho) dW
        #   dY   = Tr[Ccal(rho)] dt + dW
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = sqrt(eta) (L @ rho + rho @ Ld)

        rho = y.state
        dW = dX
        L, Lm, H = self.L(t), self.Lm(t), self.H(t)
        LdL = sum([_L.dag() @ _L for _L in L])

        # === Lcal(rho)
        # (see MESolveDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
        Hnh = -1j * H - 0.5 * LdL
        tmp = Hnh @ rho + sum([0.5 * _L @ rho @ _L.dag() for _L in L])
        Lcal_rho = tmp + dag(tmp)

        # === Ccal(rho)
        Lm_rho = stack([_Lm @ rho for _Lm in Lm])
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lm_rho + Lm_rho.dag())  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        # === state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # === measurement Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        y = cast(DiffusiveState, y)
        return DiffusiveState(rho + drho, y.Y + dY)


class DSMESolveRouchon1Integrator(
    RouchonPropertiesMixin, DSMEFixedStepIntegrator, SolveInterface
):
    """Integrator solving the diffusive SME with the Rouchon1 method."""

    method: Rouchon1

    @property
    def identity(self) -> QArray:
        return eye_like(self.H(0))

    def forward(self, t: RealScalarLike, y: SDEState, dX: Array) -> SDEState:
        # The Rouchon update for a single loss channel is:
        #   rho_{k+1} = M_dY @ rho_k @ M_dY^\dag + M1 @ rho_k @ M1d / Tr[...]
        #   dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        # with
        #   MdY = I - (iH + 0.5 Ld @ L) dt + sqrt(self.eta) * dY * Lm
        #   M1 = sqrt(1 - eta) * L sqrt(dt)
        #
        # See comment of `cholesky_normalize()` for the normalisation (computed for the
        # "average" Kraus operator).

        rho = y.state
        dW = dX
        Lc, Lm, L, H = (
            self.Lc(t + self.dt / 2),
            self.Lm(t + self.dt / 2),
            self.L(t + self.dt / 2),
            self.H(t + self.dt / 2),
        )

        # === measurement Y
        # dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        trace = jnp.stack([expect(_Lm + _Lm.dag(), rho).real for _Lm in Lm])  # (nLm)
        dY = jnp.sqrt(self.etas) * trace * self.dt + dW  # (nLm,)

        # === state rho
        M0 = self.identity - (1j * H + 0.5 * sum(_L.dag() @ _L for _L in L)) * self.dt
        if self.method.normalize:
            Ms_lindblad = [jnp.sqrt(self.dt) * _L for _L in L]
            S = M0.dag() @ M0 + sum([_M.dag() @ _M for _M in Ms_lindblad])
            rho = cholesky_normalize(S, rho)

        M_dY = M0 + sum(
            [
                jnp.sqrt(eta) * _dY * _Lm
                for eta, _dY, _Lm in zip(self.etas, dY, Lm, strict=True)
            ]
        )
        Ms = (
            [M_dY]
            + [
                jnp.sqrt((1 - eta) * self.dt) * _Lm
                for eta, _Lm in zip(self.etas, Lm, strict=True)
            ]
            + [jnp.sqrt(self.dt) * _Lc for _Lc in Lc]
        )

        rho = sum([M @ rho @ M.dag() for M in Ms])
        rho = rho / rho.trace()  # normalise by signal probability

        y = cast(DiffusiveState, y)
        return DiffusiveState(rho, y.Y + dY)


dsmesolve_euler_maruyama_integrator_constructor = DSMESolveEulerMayuramaIntegrator
dsmesolve_rouchon1_integrator_constructor = DSMESolveRouchon1Integrator
