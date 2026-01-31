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
from ...utils.general import expect
from ...utils.operators import eye_like
from .abstract_integrator import StochasticBaseIntegrator
from .interfaces import (
    DSMEInterface,
    DSSEInterface,
    JSMEInterface,
    JSSEInterface,
    SolveInterface,
)
from .rouchon_integrator import (
    KrausMap,
    MESolveFixedRouchon1Integrator,
    cholesky_normalize,
)
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

    @property
    def total_nsteps(self) -> int:
        # total number of steps of length dt
        return round((self.t1 - self.t0) / self.dt)

    @abstractmethod
    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> PyTree:
        pass

    @abstractmethod
    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        # return SDE state y_{t+dt} from a random variable sample dX and the current
        # state y_t
        pass

    @abstractmethod
    def sde_y0(self) -> SDEState:
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
        nchunks = round(nsteps / nsteps_per_chunk)

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
        # number of steps per save interval
        nsteps_per_save = round(self.total_nsteps / nsave)

        # === initial state
        # define initial SDE state
        sde_y0 = self.sde_y0()
        # save initial SDE state at time 0
        saved0 = self.save(sde_y0)

        # === run the simulation
        # integrate the SSE/SME for each save interval
        def outer_step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate_by_chunks(t, y, key, nsteps_per_save)
            return (t, y), self.save(y)

        # split the key for each save interval
        keys = jax.random.split(self.key, nsave)
        ylast, saved = jax.lax.scan(outer_step, (self.t0, sde_y0), keys)
        ylast = ylast[1]

        # === collect and format results
        # insert the initial saved result
        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        # postprocess the saved results
        saved = self.postprocess_saved(saved, ylast)

        return self.result(saved, infos=self.Infos(self.total_nsteps))


class JumpState(SDEState):
    """State for the jump SSE/SME fixed step integrators."""

    state: QArray  # state (integrated from initial to current time)
    # click indicator of shape (self.total_nsteps): 0 = no click, i + 1 = click of the
    # i-th jump operator
    clicks: Array
    step_idx: int  # step index


class JumpSolveFixedStepIntegrator(StochasticSolveFixedStepIntegrator):
    """Integrator solving the jump SSE/SME with a fixed step size integrator."""

    @property
    @abstractmethod
    def nmeas(self) -> int:
        pass

    def sde_y0(self) -> SDEState:
        clicks = jnp.full(self.total_nsteps, jnp.nan)
        return JumpState(self.y0, clicks, 0)

    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> PyTree:
        # Sample a tuple of uniform between 0 and 1 for each step. The first value is
        # used to sample a click or no click, the second value is used to sample one of
        # the jump operator when a click occurs.
        return jax.random.uniform(key, (nsteps, 2))

    def save(self, y: PyTree) -> Saved:
        return super().save(y.state)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast.state[None, :])

        # convert array of click indicators at each step to array of clicktimes, for
        # example:
        #   times = [0, 10, 20, 30, 40, 50]
        #   clicks = [0, 1, 0, 1, 0, 2]
        #   => clicktimes = [[10, 30, nan, ...], [50, nan, nan, ...]]
        clicktimes = jnp.full((self.nmeas, self.options.nmaxclick), jnp.nan)
        for jump_idx in jnp.arange(self.nmeas):
            times = self.t0 + jnp.arange(self.total_nsteps) * self.dt
            times = jnp.where(ylast.clicks == jump_idx + 1, times, jnp.nan).sort()
            ncopy = min(len(times), clicktimes.shape[1])
            clicktimes = clicktimes.at[jump_idx, :ncopy].set(times[:ncopy])

        return JumpSolveSaved(saved.ysave, saved.extra, saved.Esave, clicktimes)


class JSSESolveFixedStepIntegrator(JumpSolveFixedStepIntegrator, JSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


def _safe_divide(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    # avoid division by zero by returning 0 when y is 0
    return jnp.where(y == 0.0, 0.0, x / y)


class JSSESolveEulerJumpIntegrator(JSSESolveFixedStepIntegrator):
    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        L, H = self.L(t), self.H(t)
        L = stack(L)  # todo: remove stack
        I = eye_like(H)
        LdL = L.dag() @ L

        exp_LdL = expect(LdL, psi).real

        # === click probabilities
        tmp = exp_LdL * self.dt
        any_click_proba = sum(tmp)
        # avoid division by zero when the click probability is zero
        click_probas = _safe_divide(tmp, any_click_proba)

        # === click or no click?
        dN = jax.lax.select(dX[0] < any_click_proba, 1, 0)

        # === click
        # sample one of the jump operators
        cum_probas = jnp.cumsum(click_probas)
        jump_idx = jnp.searchsorted(
            cum_probas, dX[1], side='right', method='compare_all'
        )
        # avoid division by zero when the click probability is zero
        normalisation = _safe_divide(1, jnp.sqrt(exp_LdL[jump_idx]))
        # apply the jump operator
        psi_click = L[jump_idx] @ psi * normalisation

        # === no click
        psi_noclick = psi - (
            self.dt * (1j * H + 0.5 * sum(LdL - exp_LdL[:, None, None] * I)) @ psi
        )

        # === update state
        psi = dN * psi_click + (1 - dN) * psi_noclick

        # === update click record
        # 0 for no click, i + 1 for click of the i-th jump operator
        clicks = y.clicks.at[y.step_idx].set(dN * (jump_idx + 1))

        return JumpState(psi, clicks, y.step_idx + 1)


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

        # === click probabilities
        tmp = tr_Ccal_rho * self.dt
        any_click_proba = sum(tmp)
        # avoid division by zero when the click probability is zero
        click_probas = _safe_divide(tmp, any_click_proba)

        # === click or no click?
        dN = jax.lax.select(dX[0] < any_click_proba, 1, 0)

        # === click
        # sample one of the jump operators
        cum_probas = jnp.cumsum(click_probas)
        jump_idx = jnp.searchsorted(
            cum_probas, dX[1], side='right', method='compare_all'
        )
        # avoid division by zero when the click probability is zero
        normalisation = _safe_divide(1, tr_Ccal_rho[jump_idx])
        # apply the measurement backaction superoperator
        rho_click = Ccal_rho[jump_idx] * normalisation

        # === no click
        # compute Lcal(rho), see `MESolveDiffraxIntegrator` in
        # `integrators/core/diffrax_integrator.py`
        Hnh = -1j * H - 0.5 * sum(L.dag() @ L)
        tmp = Hnh @ rho + 0.5 * sum(Lm_rho_Lmdag)
        Lcal_rho = tmp + tmp.dag()
        rho_noclick = (
            rho
            + Lcal_rho * self.dt
            + self.dt * sum(tr_Ccal_rho[:, None, None] * rho - Ccal_rho)
        )

        # === update state
        rho = dN * rho_click + (1 - dN) * rho_noclick

        # === update click record
        # 0 for no click, i + 1 for click of the i-th jump operator
        clicks = y.clicks.at[y.step_idx].set(dN * (jump_idx + 1))

        return JumpState(rho, clicks, y.step_idx + 1)


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

    def sample_rv(self, key: PRNGKeyArray, nsteps: int) -> PyTree:
        return jnp.sqrt(self.dt) * jax.random.normal(key, (nsteps, self.nmeas))

    def save(self, y: PyTree) -> Saved:
        saved = super().save(y.state)
        return DiffusiveSolveSaved(saved.ysave, saved.extra, saved.Esave, y.Y)

    def postprocess_saved(self, saved: Saved, ylast: PyTree) -> Saved:
        saved = super().postprocess_saved(saved, ylast.state[None, :])

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

    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
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
                    for _Lpsi, _exp, _dW in zip(Lpsi, exp, dW, strict=True)
                ]
            )
        )

        return DiffusiveState(psi + dpsi, y.Y + dY)


def cholesky_normalize_ket(krausmap: KrausMap, psi: QArray) -> jax.Array:
    # See comment of `cholesky_normalize()`.
    # For a ket we compute ~M @ psi = M @ T^{†(-1)} @ psi, so we directly replace psi by
    # T^{†(-1)} @ psi.

    S = krausmap.S()
    T = jnp.linalg.cholesky(S.to_jax())  # T lower triangular

    psi = psi.to_jax()[:, 0]  # (n, 1) -> (n,)
    # solve T^† @ x = psi => x = T^{†(-1)} @ psi
    return jax.lax.linalg.triangular_solve(
        T, psi, lower=True, transpose_a=True, conjugate_a=True, left_side=True
    )[:, None]  # (n,) -> (n, 1)


class DSSESolveRouchon1Integrator(DSSEFixedStepIntegrator):
    """Integrator solving the diffusive SSE with the Rouchon1 method."""

    def forward(self, t: Scalar, y: SDEState, dX: Array) -> SDEState:
        psi = y.state
        dW = dX
        L = self.L(t)
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
        krausmap = MESolveFixedRouchon1Integrator.build_kraus_map(
            self.H, self.L, t, self.dt, self.method.time_dependent
        )
        Ms_average = krausmap.get_kraus_operators()
        if self.method.normalize:
            psi = cholesky_normalize_ket(krausmap, psi)

        M_dY = Ms_average[0] + sum([_dY * _L for _dY, _L in zip(dY, L, strict=True)])

        psi = (M_dY @ psi).unit()  # normalise by signal probability

        return DiffusiveState(psi, y.Y + dY)


dssesolve_euler_maruyama_integrator_constructor = DSSESolveEulerMayuramaIntegrator
dssesolve_rouchon1_integrator_constructor = DSSESolveRouchon1Integrator


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
        # (see MESolveDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
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
        # "average" Kraus operator).

        rho = y.state
        dW = dX
        Lc, Lm = self.Lc(t), self.Lm(t)

        # === measurement Y
        # dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        trace = jnp.stack([expect(_Lm + _Lm.dag(), rho).real for _Lm in Lm])  # (nLm)
        dY = jnp.sqrt(self.etas) * trace * self.dt + dW  # (nLm,)

        # === state rho
        kraus_map = MESolveFixedRouchon1Integrator.build_kraus_map(
            self.H, self.L, t, self.dt, self.method.time_dependent
        )
        Ms_average = kraus_map.get_kraus_operators()
        if self.method.normalize:
            rho = cholesky_normalize(kraus_map, rho)

        M_dY = Ms_average[0] + sum(
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
            + [self.dt * _Lc for _Lc in Lc]
        )

        rho = sum([M @ rho @ M.dag() for M in Ms])
        rho = rho / rho.trace()  # normalise by signal probability

        return DiffusiveState(rho, y.Y + dY)


dsmesolve_euler_maruyama_integrator_constructor = DSMESolveEulerMayuramaIntegrator
dsmesolve_rouchon1_integrator_constructor = DSMESolveRouchon1Integrator
