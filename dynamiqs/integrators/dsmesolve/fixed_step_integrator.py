from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Scalar

from ...result import Result
from ...utils.quantum_utils.general import dag, trace
from ..core.abstract_integrator import DSMESolveIntegrator
from ..core.save_mixin import DSMESolveSaveMixin


class YDSME(eqx.Module):
    """State for the diffusive SME fixed step integrators."""

    rho: Array  # state (integrated from initial to current time)
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

        # === Lcal(rho)
        # (see MEDiffraxIntegrator in `integrators/core/diffrax_integrator.py`)
        L = self.L(t)
        Ld = dag(L)
        LdL = (Ld @ L).sum(0)
        tmp = (-1j * self.H(t) - 0.5 * LdL) @ y.rho + 0.5 * (L @ y.rho @ Ld).sum(0)
        Lcal_rho = tmp + dag(tmp)

        # === Ccal(rho)
        Lm = self.Lm(t)  # (nLm, n, n)
        Lm_rho = Lm @ y.rho
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lm_rho + dag(Lm_rho))  # (nLm, n, n)
        tr_Ccal_rho = trace(Ccal_rho).real  # (nLm,)

        # === state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * y.rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # === measurement Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        return YDSME(drho, dY)
