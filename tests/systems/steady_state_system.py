from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.gradient import Gradient
from dynamiqs.method import DiffraxSteadyState, Kvaerno5, Method
from dynamiqs.qarrays.layout import Layout
from dynamiqs.result import MESteadyStateResult


class DrivenThermalCavity:
    # driven cavity coupled to a thermal bath: its unique steady state is the displaced
    # thermal state D(alpha) rho_th(nth) D(alpha)^†, whose purity depends on nth
    class Params(NamedTuple):
        delta: float
        eps: float
        kappa: float
        nth: float

    def __init__(
        self,
        *,
        n: int,
        delta: float,
        eps: float,
        kappa: float,
        nth: float,
        layout: Layout,
    ):
        self.n = n
        self.layout = layout

        # define default gradient parameters
        self.params_default = self.Params(delta, eps, kappa, nth)

    def H(self, params: PyTree) -> QArray:
        a = dq.destroy(self.n, layout=self.layout)
        return params.delta * a.dag() @ a + params.eps * (a + a.dag())

    def Ls(self, params: PyTree) -> list[QArray]:
        a = dq.destroy(self.n, layout=self.layout)
        return [
            jnp.sqrt(params.kappa * (1 + params.nth)) * a,
            jnp.sqrt(params.kappa * params.nth) * a.dag(),
        ]

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [
            dq.position(self.n, layout=self.layout),
            dq.momentum(self.n, layout=self.layout),
            dq.number(self.n, layout=self.layout),
        ]

    def run(
        self,
        # few steps, so that a stop that never fires fails fast
        method: Method = DiffraxSteadyState(Kvaerno5(max_steps=1000)),  # noqa: B008
        *,
        gradient: Gradient | None = None,
        params: PyTree | None = None,
    ) -> MESteadyStateResult:
        params = self.params_default if params is None else params
        return dq.mesteadystate(
            self.H(params),
            self.Ls(params),
            exp_ops=self.Es(params),
            method=method,
            gradient=gradient,
        )

    def _alpha(self, params: PyTree) -> Array:
        return -1j * params.eps / (1j * params.delta + 0.5 * params.kappa)

    def state(self) -> QArray:
        D = dq.displace(self.n, self._alpha(self.params_default))
        return D @ dq.thermal_dm(self.n, self.params_default.nth) @ D.dag()

    def _expect(self, params: PyTree) -> Array:
        # analytical expectation values (<x>, <p>, <a^†a>) as a function of the
        # parameters
        alpha = self._alpha(params)
        return jnp.array([alpha.real, alpha.imag, jnp.abs(alpha) ** 2 + params.nth])

    def expect(self) -> Array:
        return self._expect(self.params_default)

    def grads_expect(self) -> PyTree:
        return jax.jacrev(self._expect)(self.params_default)


dense_thermal_cavity = DrivenThermalCavity(
    n=16, delta=0.7, eps=0.4, kappa=1.0, nth=0.2, layout=dq.dense
)
dia_thermal_cavity = DrivenThermalCavity(
    n=16, delta=0.7, eps=0.4, kappa=1.0, nth=0.2, layout=dq.dia
)
