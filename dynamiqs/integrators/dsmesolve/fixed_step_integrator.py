from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Scalar

from ...qarrays.qarray import QArray
from ...qarrays.utils import stack, sum_qarrays
from ...result import Result
from ..core.abstract_integrator import DSMESolveIntegrator
from ..core.save_mixin import DSMESolveSaveMixin


class YDSME(eqx.Module):
    """State for the diffusive SME fixed step integrators."""

    rho: QArray  # state (integrated from initial to current time)
    Y: Array  # measurement (integrated from initial to current time)

    def __add__(self, other: Any) -> YDSME:
        return YDSME(self.rho + other.rho, self.Y + other.Y)


class DSMEFixedStepIntegrator(DSMESolveIntegrator, DSMESolveSaveMixin):
    """Integrator solving the diffusive SME with a fixed step size integrator."""

    def __check_init__(self):
        # todo: check tsave elements align correctly with dt
        pass

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

    def run(self) -> Result:
        # === define variables
        nLm = len(self.etas)
        # number of save interval
        nsave = len(self.ts) - 1
        # number of steps (of length dt)
        nsteps = int((self.t1 - self.t0) / self.dt)
        # number of steps per save interval
        nsteps_per_save = int(nsteps // nsave)
        # save time
        delta_t = self.ts[1] - self.ts[0]

        # === initial state
        # define initial state (rho, Y) = (rho0, 0)
        y0 = YDSME(self.y0, jnp.zeros(nLm))
        # save initial state at time 0
        saved0 = self.save(y0)

        # === run the simulation
        # The run section encapsulates two loop. The outer loop iterates over each time
        # in `self.ts` to regularly save the integrated state rho and measurement Y. The
        # inner loop iterates over the fixed step size dt.

        # iterate over each save interval
        def outer_step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry

            # sample wiener
            # note: we could do this once outside the outer loop, but doing it here
            # prevents memory overload
            dWs = jnp.sqrt(self.dt) * jax.random.normal(key, (nsteps_per_save, nLm))

            # iterate over the fixed step size dt
            def step(carry, dW):  # noqa: ANN001, ANN202
                t, y = carry
                y = y + self.forward(t, y, dW)
                t = t + self.dt
                return (t, y), None

            (t, y), _ = jax.lax.scan(step, (t, y), dWs)

            return (t, y), self.save(y)

        # split the key to generate wiener on each save interval
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
    def forward(self, t: Scalar, y: YDSME, dW: Array) -> YDSME:
        # return (drho, dY)
        pass


class DSMESolveEulerMayuramaIntegrator(DSMEFixedStepIntegrator):
    """Integrator solving the diffusive SME with the Euler-Mayurama method."""

    def forward(self, t: Scalar, y: YDSME, dW: Array) -> YDSME:
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
        Lm_rho = stack([_L @ y.rho for _L in Lm])
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lm_rho + Lm_rho.dag())  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        # === state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * y.rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # === measurement Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        return YDSME(drho, dY)
