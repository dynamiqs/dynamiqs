from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from dynamiqs import basis, dag, sigmaz
from dynamiqs.gradient import Gradient
from dynamiqs.integrators.apis.floquet import floquet
from dynamiqs.options import Options
from dynamiqs.result import FloquetResult
from dynamiqs.solver import Solver
from dynamiqs.time_array import CallableTimeArray, timecallable

from ..system import System


class FloquetQubit(System):
    class Params(NamedTuple):
        omega: float
        omega_d: float
        amp: float
        tsave: Array
        modes_0: Array | None
        quasienergies: Array | None

    def run(
        self,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: Options = Options(),  # noqa: B008
        params: PyTree | None = None,
    ) -> FloquetResult:
        params = self.params_default if params is None else params
        H = self.H(params)
        T = 2.0 * jnp.pi / params.omega_d
        return floquet(
            H, T, params.tsave, solver=solver, gradient=gradient, options=options
        )

    def __init__(
        self,
        omega: float,
        omega_d: float,
        amp: float,
        tsave: Array,
        *,
        modes_0: Array | None = None,
        quasienergies: Array | None = None,
    ):
        self.omega = omega
        self.omega_d = omega_d
        self.amp = amp
        self.tsave = tsave
        self.modes_0 = modes_0
        self.quasienergies = quasienergies

        self.params_default = self.Params(
            omega, omega_d, amp, tsave, modes_0, quasienergies
        )

    def H(self, params: PyTree) -> CallableTimeArray:
        sigmap = basis(2, 1) @ dag(basis(2, 0))

        def H_func(t):
            H0 = -0.5 * params.omega * sigmaz()
            H1 = 0.5 * params.amp * sigmap * jnp.exp(-1j * params.omega_d * t)
            return H0 + H1 + dag(H1)

        return timecallable(H_func)

    def state(self, t: float) -> Array:
        delta_Omega = self.omega - self.omega_d
        theta = jnp.arctan(self.amp / delta_Omega)
        w0 = jnp.cos(0.5 * theta) * basis(2, 0) - jnp.exp(
            -1j * self.omega_d * t
        ) * jnp.sin(0.5 * theta) * basis(2, 1)
        w1 = jnp.sin(0.5 * theta) * basis(2, 0) + jnp.exp(
            -1j * self.omega_d * t
        ) * jnp.cos(0.5 * theta) * basis(2, 1)
        return jnp.stack([w0, w1])

    def true_quasienergies(self) -> Array:
        delta_Omega = self.omega - self.omega_d
        Omega_R = jnp.sqrt(delta_Omega**2 + self.amp**2)
        quasi_es = jnp.asarray([0.5 * Omega_R, -0.5 * Omega_R])
        quasi_es = jnp.mod(quasi_es, self.omega_d)
        return jnp.where(
            quasi_es > 0.5 * self.omega_d, quasi_es - self.omega_d, quasi_es
        )

    def y0(self, params: PyTree) -> Array:
        raise NotImplementedError

    def Es(self, params: PyTree) -> Array:
        raise NotImplementedError