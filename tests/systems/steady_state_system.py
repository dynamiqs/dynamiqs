from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.qarrays.layout import Layout
from dynamiqs.result import Result

from ._system import System


class DrivenOCavity(System):
    # H = Δ a†a + ε (a + a†), L = √κ a: the steady state is the coherent state
    # |alpha⟩ with alpha = -iε / (iΔ + κ/2)
    class Params(NamedTuple):
        eps: float
        delta: float
        kappa: float

    def __init__(
        self, *, n: int, eps: float, delta: float, kappa: float, layout: Layout
    ):
        self.n = n
        self.eps = eps
        self.delta = delta
        self.kappa = kappa
        self.layout = layout
        self.params_default = self.Params(eps, delta, kappa)

    def H(self, params: PyTree) -> QArray:
        a = dq.destroy(self.n, layout=self.layout)
        return params.delta * a.dag() @ a + params.eps * (a + a.dag())

    def Ls(self, params: PyTree) -> list[QArray]:
        return [jnp.sqrt(params.kappa) * dq.destroy(self.n, layout=self.layout)]

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.fock(self.n, 0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.destroy(self.n, layout=self.layout)]

    def _alpha(self) -> Array:
        return -1j * self.eps / (1j * self.delta + 0.5 * self.kappa)

    def steady_state(self) -> QArray:
        return dq.coherent_dm(self.n, self._alpha())

    def steady_expect(self) -> Array:
        return jnp.array([self._alpha()])

    def loss_state(self, state: QArray) -> Array:
        return dq.expect(dq.number(self.n, layout=self.layout), state).real

    def steady_grads_state(self) -> PyTree:
        # ⟨a†a⟩ = |alpha|² = ε² / D with D = Δ² + κ²/4
        D = self.delta**2 + 0.25 * self.kappa**2
        grad_eps = 2 * self.eps / D
        grad_delta = -2 * self.eps**2 * self.delta / D**2
        grad_kappa = -0.5 * self.eps**2 * self.kappa / D**2
        return self.Params(grad_eps, grad_delta, grad_kappa)

    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        params: PyTree | None = None,
        vectorized: bool = False,
    ) -> Result:
        params = self.params_default if params is None else params
        return dq.steadystate(
            self.H(params),
            self.Ls(params),
            exp_ops=self.Es(params),
            method=method,
            gradient=gradient,
            vectorized=vectorized,
        )


# |alpha|² = 0.2, far from the truncation at n = 16
dense_driven_ocavity = DrivenOCavity(
    n=16, eps=0.5, delta=1.0, kappa=1.0, layout=dq.dense
)
dia_driven_ocavity = DrivenOCavity(n=16, eps=0.5, delta=1.0, kappa=1.0, layout=dq.dia)
