from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray, Scalar

from ...qarrays.qarray import QArray
from ...qarrays.utils import stack, sum_qarrays
from ...result import Result
from .abstract_integrator import StochasticBaseIntegrator
from .interfaces import DSMEInterface, SolveInterface
from .save_mixin import DSMESolveSaveMixin


class DSMEState(eqx.Module):
    """State for the diffusive SME fixed step integrators."""

    rho: QArray  # state (integrated from initial to current time)
    Y: Array  # measurement (integrated from initial to current time)

    def __add__(self, other: Any) -> DSMEState:
        return DSMEState(self.rho + other.rho, self.Y + other.Y)


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


class DSMEFixedStepIntegrator(
    StochasticBaseIntegrator, DSMEInterface, DSMESolveSaveMixin
):
    """Integrator solving the diffusive SME with a fixed step size integrator."""

    def __check_init__(self):
        # check that all tsave values are exact multiples of dt
        if not _is_multiple_of(self.ts, self.dt):
            raise ValueError(
                'Argument `tsave` should only contain exact multiples of the solver '
                'fixed step size `dt`.'
            )

        # check that tsave is linearly spaced
        if not _is_linearly_spaced(self.ts):
            raise ValueError('Argument `tsave` should be linearly spaced.')

        # check that options.t0 is not used
        if self.options.t0 is not None:
            raise ValueError('Option `t0` is invalid for fixed step SME solvers.')

        if self.discontinuity_ts is not None:
            warnings.warns(
                'The Hamiltonian or jump operators are time-dependent with '
                'discontinuities, which will be ignored by the solver.',
                stack_level=1,
            )

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    @property
    def dt(self) -> float:
        return self.solver.dt

    def integrate(
        self, t0: float, y0: DSMEState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, DSMEState]:
        # integrate the SME for nsteps of length dt
        nLm = len(self.etas)

        # sample wiener
        dWs = jnp.sqrt(self.dt) * jax.random.normal(key, (nsteps, nLm))

        # iterate over the fixed step size dt
        def step(carry, dW):  # noqa: ANN001, ANN202
            t, y = carry
            y = y + self.forward(t, y, dW)
            t = t + self.dt
            return (t, y), None

        (t, y), _ = jax.lax.scan(step, (t0, y0), dWs)

        return t, y

    def integrate_by_chunks(
        self, t0: float, y0: DSMEState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, DSMEState]:
        # integrate the SME for nsteps of length dt, splitting the integration in
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
        nLm = len(self.etas)
        # number of save interval
        nsave = len(self.ts) - 1
        # total number of steps (of length dt)
        nsteps = int((self.t1 - self.t0) / self.dt)
        # number of steps per save interval
        nsteps_per_save = int(nsteps // nsave)
        # save time
        delta_t = self.ts[1] - self.ts[0]

        # === initial state
        # define initial state (rho, Y) = (rho0, 0)
        y0 = DSMEState(self.y0, jnp.zeros(nLm))
        # save initial state at time 0
        saved0 = self.save(y0)

        # === run the simulation
        # integrate the SME for each save interval
        def outer_step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate_by_chunks(t, y, key, nsteps_per_save)
            return (t, y), self.save(y)

        # split the key for each save interval
        keys = jax.random.split(self.key, nsave)
        ylast, saved = jax.lax.scan(outer_step, (self.t0, y0), keys)
        ylast = ylast[1]

        # === collect and format results
        # insert the initial saved result
        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        # The averaged measurement I^{(ta, tb)} is recovered by diffing the measurement
        # Y which is integrated between ts[0] and ts[-1].
        Isave = jnp.diff(saved.Isave, axis=0) / delta_t
        saved = eqx.tree_at(lambda x: x.Isave, saved, Isave)
        saved = self.postprocess_saved(saved, ylast)

        return self.result(saved, infos=self.Infos(nsteps))

    @abstractmethod
    def forward(self, t: Scalar, y: DSMEState, dW: Array) -> DSMEState:
        # return (drho, dY)
        pass


class DSMESolveEulerMayuramaIntegrator(DSMEFixedStepIntegrator, SolveInterface):
    """Integrator solving the diffusive SME with the Euler-Mayurama method."""

    def forward(self, t: Scalar, y: DSMEState, dW: Array) -> DSMEState:
        # The diffusive SME for a single detector is:
        #   drho = Lcal(rho)     dt + (Ccal(rho) - Tr[Ccal(rho)] rho) dW
        #   dY   = Tr[Ccal(rho)] dt + dW
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = sqrt(eta) (L @ rho + rho @ Ld)
        L, Lm, H = self.L(t), self.Lm(t), self.H(t)

        # === Lcal(rho)
        # (see MEDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
        Hnh = sum_qarrays(-1j * H, *[-0.5 * _L.dag() @ _L for _L in L])
        tmp = sum_qarrays(Hnh @ y.rho, *[0.5 * _L @ y.rho @ _L.dag() for _L in L])
        Lcal_rho = tmp + tmp.dag()

        # === Ccal(rho)
        Lm_rho = stack([_Lm @ y.rho for _Lm in Lm])
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lm_rho + Lm_rho.dag())  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        # === state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * y.rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # === measurement Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        return DSMEState(drho, dY)


dsmesolve_euler_maruyama_integrator_constructor = DSMESolveEulerMayuramaIntegrator
