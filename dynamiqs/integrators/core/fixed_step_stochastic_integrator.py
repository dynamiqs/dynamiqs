from __future__ import annotations

import warnings
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar

from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import DiffusiveSolveSaved, JumpSolveSaved, Result, Saved
from ...utils.general import dag, expect
from ...utils.operators import eye_like
from .abstract_integrator import StochasticBaseIntegrator
from .interfaces import (
    DSMEInterface,
    DSSEInterface,
    JSMEInterface,
    JSSEInterface,
    SolveInterface,
)
from .rouchon_integrator import cholesky_normalize
from .save_mixin import SolveSaveMixin


def _is_multiple_of(
    x: ArrayLike, dt: float, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    x_rounded = np.round(np.array(x) / dt) * dt
    return np.allclose(x, x_rounded, rtol=rtol, atol=atol)


def _is_linearly_spaced(
    x: ArrayLike, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    diffs = np.diff(x)
    return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)


class SDEState(eqx.Module):
    """State for the jump/diffusive SSE/SME fixed step integrators."""


class StochasticSolveFixedStepIntegrator(
    StochasticBaseIntegrator, SolveInterface, SolveSaveMixin
):
    """Integrator solving the jump/diffusive SSE/SME with a fixed step size
    integrator.
    """

    def __check_init__(self):
        # check that all tsave values are exact multiples of dt
        if not _is_multiple_of(self.ts, self.dt):
            raise ValueError(
                'Argument `tsave` should only contain exact multiples of the method '
                'fixed step size `dt`.'
            )

        # check that tsave is linearly spaced
        if not _is_linearly_spaced(self.ts):
            raise ValueError('Argument `tsave` should be linearly spaced.')

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
        return self.method.dt

    @abstractmethod
    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> Array:
        pass

    @abstractmethod
    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        # return SDE state y_{t+dt} from a random variable sample dX and the current
        # state y_t
        pass

    @abstractmethod
    def sde_y0(self, key: PRNGKeyArray) -> SDEState:
        # define initial SDE state
        pass

    def integrate(
        self, t0: float, y0: SDEState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, SDEState]:
        # integrate the SDE for nsteps of length dt

        # sample random variable driving the SME
        dXs = self.sample_rv(key, nsteps)

        # iterate over the fixed step size dt
        def step(carry, dX):  # noqa: ANN001, ANN202
            t, y = carry
            y = self.forward(t, y, dX)
            t = t + self.dt
            return (t, y), None

        (t, y), _ = jax.lax.scan(step, (t0, y0), dXs)

        return t, y

    def integrate_by_chunks(
        self, t0: float, y0: SDEState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, SDEState]:
        # integrate the SDE for nsteps of length dt, splitting the integration in
        # chunks of 1000 dts to ensure a fixed memory usage

        nsteps_per_chunk = 1000
        nchunks = int(nsteps // nsteps_per_chunk)

        # iterate over each chunk
        def step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate(t, y, key, nsteps_per_chunk)
            return (t, y), None

        # split the key for each chunk
        key, lastkey = jax.random.split(key)
        keys = jax.random.split(key, (nchunks,))
        (t, y), _ = jax.lax.scan(step, (t0, y0), keys)

        # integrate for the remaining number of steps (< nsubsteps)
        nremaining = nsteps % nsteps_per_chunk
        t, y = self.integrate(t, y, lastkey, nremaining)

        return t, y

    def run(self) -> Result:
        # this function assumes that ts is linearly spaced with each value an exact
        # multiple of dt

        # === define variables
        # number of save interval
        nsave = len(self.ts) - 1
        # total number of steps (of length dt)
        nsteps = int((self.t1 - self.t0) / self.dt)
        # number of steps per save interval
        nsteps_per_save = int(nsteps // nsave)

        # === initial state
        # define initial SDE state
        key0, key = jax.random.split(self.key)
        sde_y0 = self.sde_y0(key0)
        # save initial SDE state at time 0
        saved0 = self.save(sde_y0)

        # === run the simulation
        # integrate the SSE/SME for each save interval
        def outer_step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate_by_chunks(t, y, key, nsteps_per_save)
            return (t, y), self.save(y)

        # split the key for each save interval
        keys = jax.random.split(key, nsave)
        ylast, saved = jax.lax.scan(outer_step, (self.t0, sde_y0), keys)
        ylast = ylast[1]

        # === collect and format results
        # insert the initial saved result
        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        # postprocess the saved results
        saved = self.postprocess_saved(saved, ylast)

        return self.result(saved, infos=self.Infos(nsteps))


class JumpState(SDEState):
    """State for the jump SSE/SME fixed step integrators."""

    state: QArray  # state (integrated from initial to current time)
    clicktimes: Array  # click times of shape (nLs, nmaxclick)
    indices: Array  # last click time indices of shape (nLs,)
    key_jump_choice: PRNGKeyArray  # random key to sample a click jump operator

    def new_click(self, idx: int, t: float) -> tuple[Array, Array]:
        clicktimes = self.clicktimes.at[idx, self.indices[idx]].set(t)
        indices = self.indices.at[idx].add(1)
        return clicktimes, indices


class JumpSolveFixedStepIntegrator(StochasticSolveFixedStepIntegrator):
    """Integrator solving the jump SSE/SME with a fixed step size integrator."""

    @property
    @abstractmethod
    def nmeas(self) -> int:
        pass

    def sde_y0(self, key: PRNGKeyArray) -> SDEState:
        clicktimes = jnp.full((self.nmeas, self.options.nmaxclick), jnp.nan)
        indices = jnp.zeros(self.nmeas, dtype=int)
        key_jump_choice = key
        return JumpState(self.y0, clicktimes, indices, key_jump_choice)

    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> Array:
        return jax.random.uniform(key, nsteps)  # sample uniform between 0 and 1

    def save(self, y: PyTree) -> Saved:
        return super().save(y.state)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast.state)
        return JumpSolveSaved(saved.ysave, saved.extra, saved.Esave, ylast.clicktimes)


class JSSESolveFixedStepIntegrator(JumpSolveFixedStepIntegrator, JSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


class JSSESolveEulerJumpIntegrator(JSSESolveFixedStepIntegrator):
    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        L, H = self.L(t), self.H(t)
        L = stack(L)  # todo: remove stack
        I = eye_like(H)
        LdL = L.dag() @ L

        exp_LdL = expect(LdL, psi).real
        tmp = exp_LdL * self.dt
        any_click_proba = sum(tmp)
        L_probas = tmp / any_click_proba

        def click(y: JumpState) -> JumpState:
            # sample one of the jump operators
            key, newkey = jax.random.split(y.key_jump_choice)
            idx = jax.random.choice(key, self.nmeas, p=L_probas)

            # apply the jump operator
            psi = L[idx] @ y.state / jnp.sqrt(exp_LdL[idx])

            # update click time and index
            clicktimes, indices = y.new_click(idx, t)

            return JumpState(psi, clicktimes, indices, newkey)

        def noclick(y: JumpState) -> JumpState:
            psi = y.state
            psi -= (
                self.dt * (1j * H + 0.5 * sum(LdL - exp_LdL[:, None, None] * I)) @ psi
            )
            return eqx.tree_at(lambda y: y.state, y, psi)

        return jax.lax.cond(dX < any_click_proba, click, noclick, y)


jssesolve_euler_jump_integrator_constructor = JSSESolveEulerJumpIntegrator


class JSMESolveFixedStepIntegrator(JumpSolveFixedStepIntegrator, JSMEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.etas)


class JSMESolveEulerJumpIntegrator(JSMESolveFixedStepIntegrator):
    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
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
        Lm = stack(self.Lm(t))

        # === Ccal(rho)
        Lm_rho_Lmdag = Lm @ rho @ Lm.dag()  # (nLm, n, n)
        thetas = self.thetas[:, None, None]  # (nLm, 1, 1)
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = thetas * rho + etas * Lm_rho_Lmdag  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        click_probas = tr_Ccal_rho * self.dt
        any_click_proba = sum(click_probas)

        def click(y):  # noqa: ANN001, ANN202
            # sample one of the jump operators
            key, newkey = jax.random.split(y.key_jump_choice)
            idx = jax.random.choice(key, self.nmeas, p=click_probas / any_click_proba)

            # apply the measurement backaction superoperator
            rho = Ccal_rho[idx] / tr_Ccal_rho[idx]

            # update click time and index
            clicktimes, indices = y.new_click(idx, t)

            return JumpState(rho, clicktimes, indices, newkey)

        def noclick(y):  # noqa: ANN001, ANN202
            rho = y.state

            # === Lcal(rho)
            # (see MEDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
            Hnh = -1j * H - 0.5 * sum(L.dag() @ L)
            tmp = Hnh @ rho + 0.5 * sum(Lm_rho_Lmdag)
            Lcal_rho = tmp + tmp.dag()

            rho += Lcal_rho * self.dt + self.dt * sum(
                tr_Ccal_rho[:, None, None] * rho - Ccal_rho
            )
            return eqx.tree_at(lambda y: y.state, y, rho)

        return jax.lax.cond(dX < any_click_proba, click, noclick, y)


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

    def sde_y0(self, key: PRNGKeyArray) -> SDEState:  # noqa: ARG002
        # define initial SDE state (state, Y) = (state0, 0)
        return DiffusiveState(self.y0, jnp.zeros(self.nmeas))

    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> Array:
        return jnp.sqrt(self.dt) * jax.random.normal(key, (nsteps, self.nmeas))

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y.state)
        return DiffusiveSolveSaved(saved.ysave, saved.extra, saved.Esave, y.Y)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast.state)

        # The averaged measurement I^{(ta, tb)} is recovered by diffing the measurement
        # I which is integrated between ts[0] and ts[-1]
        delta_t = self.ts[1] - self.ts[0]
        Isave = jnp.diff(saved.Isave, axis=0) / delta_t

        # reorder Isave after jax.lax.scan stacking (ntsave, nLm) -> (nLm, ntsave)
        Isave = Isave.swapaxes(-1, -2)
        return eqx.tree_at(lambda x: x.Isave, saved, Isave)


class DSSEFixedStepIntegrator(DiffusiveSolveFixedStepIntegrator, DSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


class DSSESolveEulerMayuramaIntegrator(DSSEFixedStepIntegrator):
    """Integrator solving the diffusive SSE with the Euler-Mayurama method."""

    def forward_(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
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
        exp = jnp.stack(
            [2 * (psi.dag() @ _Lpsi).squeeze((-1, -2)).real for _Lpsi in Lpsi]
        )  # (nL)
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
                    for _L, _Lpsi, _exp, _dW in zip(L, Lpsi, exp, dW, strict=True)
                ]
            )
        )

        return DiffusiveState(psi + dpsi, y.Y + dY)

    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        dW = dX
        L, H = self.L(t), self.H(t)
        I = eye_like(H)
        Lpsi = [_L @ psi for _L in L]
        LdL = sum([_L.dag() @ _L for _L in L])

        # === measurement Y
        # dY = <L+Ld> dt + dW
        # small trick to compute <L+Ld>
        #   <L+Ld> = <psi|L+Ld|psi >
        #          = psi.dag() @ (L + Ld) @ psi
        #          = psi.dag() @ L @ psi + (psi.dag() @ L @ psi).dag()
        #          = 2 Re[psi.dag() @ Lpsi]
        exp = jnp.stack(
            [2 * (psi.dag() @ _Lpsi).squeeze((-1, -2)).real for _Lpsi in Lpsi]
        )  # (nL)
        dY = exp * self.dt + dW

        # === state psi
        M0 = I - (1j * H + 0.5 * LdL) * self.dt
        M_dY = M0 + sum([_dY * _L for _dY, _L in zip(dY, L, strict=True)])

        S = M0.dag() @ M0 + (LdL * self.dt if LdL != 0.0 else 0.0)
        T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular
        M_dY = M_dY @ jnp.linalg.inv(T).mT.conj()  # M_dY @ Td^{-1}

        psi = (M_dY @ psi).unit()

        return DiffusiveState(psi, y.Y + dY)


dssesolve_euler_maruyama_integrator_constructor = DSSESolveEulerMayuramaIntegrator


class DSMEFixedStepIntegrator(DiffusiveSolveFixedStepIntegrator, DSMEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.etas)


class DSMESolveEulerMayuramaIntegrator(DSMEFixedStepIntegrator):
    """Integrator solving the diffusive SME with the Euler-Mayurama method."""

    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
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
        # (see MEDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
        Hnh = -1j * H - 0.5 * LdL
        tmp = Hnh @ rho + sum([0.5 * _L @ rho @ _L.dag() for _L in L])
        Lcal_rho = tmp + tmp.dag()

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

        return DiffusiveState(rho + drho, y.Y + dY)


class DSMESolveRouchon1Integrator(DSMEFixedStepIntegrator, SolveInterface):
    """Integrator solving the diffusive SME with the Rouchon1 method."""

    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        # The Rouchon update for a single loss channel is:
        #   rho_{k+1} = M_dY @ rho_k @ M_dY^\dag + M1 @ rho_k @ M1d / Tr[...]
        #   dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        # with
        #   MdY = I - (iH + 0.5 Ld @ L) dt + sqrt(self.eta) * dY * Lm
        #   M1 = sqrt(1 - eta) * L sqrt(dt)
        #
        # See comment of `cholesky_normalize()` for the normalisation (computed for the
        # "average" Kraus operators M0 = I - (iH + 0.5 Ld @ L) dt and M1 = L sqrt(dt)).

        rho = y.state
        dW = dX
        L, Lc, Lm, H = self.L(t), self.Lc(t), self.Lm(t), self.H(t)
        I = eye_like(H)
        LdL = sum([_L.dag() @ _L for _L in L])

        # === measurement Y
        # dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        trace = jnp.stack([expect(_Lm + _Lm.dag(), rho).real for _Lm in Lm])  # (nLm)
        dY = jnp.sqrt(self.etas) * trace * self.dt + dW  # (nLm,)

        # === state rho
        M0 = I - (1j * H + 0.5 * LdL) * self.dt
        M_dY = M0 + sum(
            [
                jnp.sqrt(eta) * _dY * _Lm
                for eta, _dY, _Lm in zip(self.etas, dY, Lm, strict=True)
            ]
        )
        Ms = [
            jnp.sqrt((1 - eta) * self.dt) * _Lm
            for eta, _Lm in zip(self.etas, Lm, strict=True)
        ] + [self.dt * _Lc for _Lc in Lc]

        if self.method.normalize:
            rho = cholesky_normalize(M0, LdL, self.dt, rho)

        rho = M_dY @ rho @ dag(M_dY) + sum([_M @ rho @ dag(_M) for _M in Ms])
        rho = rho / rho.trace()  # normalise by signal probability

        return DiffusiveState(rho, y.Y + dY)


dsmesolve_euler_maruyama_integrator_constructor = DSMESolveEulerMayuramaIntegrator
dsmesolve_rouchon1_integrator_constructor = DSMESolveRouchon1Integrator
