from __future__ import annotations

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


# state for the fixed step integrator for diffusive SMEs
class YDSME(eqx.Module):
    # The SME state at each time is (rho, Y) with rho the state (integrated from initial
    # to current time) and the signal Y (integrated from initial to current time).

    rho: Array
    Y: Array

    def __add__(self, other: Any) -> YDSME:
        return YDSME(self.rho + other.rho, self.Y + other.Y)


class DSMEFixedStepIntegrator(DSMESolveIntegrator, DSMESolveSaveMixin):
    """Integrator solving the diffusive SME with a homemade fixed step size solver."""

    def __check_init__(self):
        # todo: check dt align with discontinuity_ts and tsave
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
        nLm = len(self.etas)

        y0 = YDSME(self.y0, jnp.zeros(nLm))

        # save initial state at time 0
        # 0    1    2
        # +----+----+
        # nsave = 3
        # nintervals = 2
        saved0 = self.save(y0)
        nsave = len(self.ts)
        ndt = int((self.t1 - self.t0) / self.dt)

        # number of dt per save interval
        ndt_per_save = ndt // (nsave - 1)

        def outer_step(carry, key):  # noqa: ANN001, ANN202
            # avoid sampling crazy amount of wiener
            t, y = carry

            # sample wiener ~ N(0, dt)
            dWs = jnp.sqrt(self.dt) * jax.random.normal(key, (ndt_per_save, nLm))

            def step(carry, dW):  # noqa: ANN001, ANN202
                t, y = carry
                y = y + self.forward(t, y, dW)
                t = t + self.dt
                return (t, y), None

            (t, y), _ = jax.lax.scan(step, (t, y), dWs)

            return (t, y), self.save(y)

        keys = jax.random.split(self.key, nsave - 1)
        ylast, saved = jax.lax.scan(outer_step, (self.t0, y0), keys)
        ylast = ylast[1]

        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        # We integrate the state YSME from t0 to t1. In this case, the state is
        # (rho, Y). So we recover the signal J^{(t0, t1)} by simply diffing the
        # resulting Y array.
        Isave = jnp.diff(saved.Isave, axis=0) / self.dt
        saved = eqx.tree_at(lambda x: x.Isave, saved, Isave)
        saved = self.postprocess_saved(saved, ylast)

        return self.result(saved, infos=self.Infos(ndt))

    def forward(self, t: Scalar, y: YDSME, dW: Array) -> YDSME:
        pass


class DSMESolveEulerIntegrator(DSMEFixedStepIntegrator):
    def forward(self, t: Scalar, y: YDSME, dW: Array) -> YDSME:
        # The diffusive SME is a coupled system of SDEs with state (rho, Y), which for
        # a single detector writes:
        #   drho = Lcal(rho)     dt + (Ccal(rho) - Tr[Ccal(rho)] rho) dW
        #   dY   = Tr[Ccal(rho)] dt + dW
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = sqrt(eta) (L @ rho + rho @ Ld)

        # Lcal(rho) (see MEDiffraxIntegrator)
        Ls = jnp.stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        LdL = (Lsd @ Ls).sum(0)
        H = self.H(t)
        tmp = (-1j * H - 0.5 * LdL) @ y.rho + 0.5 * (Ls @ y.rho @ Lsd).sum(0)
        Lcal_rho = tmp + dag(tmp)

        # Ccal(rho)
        Lms = jnp.stack([L(t) for L in self.Lms])  # (nLm, n, n)
        Lms_rho = Lms @ y.rho
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lms_rho + dag(Lms_rho))  # (nLm, n, n)
        tr_Ccal_rho = trace(Ccal_rho).real  # (nLm,)

        # state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * y.rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # signal Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        return YDSME(drho, dY)
