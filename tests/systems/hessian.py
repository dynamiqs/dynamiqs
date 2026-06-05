from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree, ScalarLike

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray

from ._system import System


class HQubit(System):
    class Params(NamedTuple):
        w: float

    def __init__(self, *, w: float, tsave: Array):
        self.n = 2
        self.w = w
        self.tsave = tsave
        self.params_default = self.Params(w)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        return 0.5 * params.w * dq.sigmax()

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.basis(2, 0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.sigmaz()]

    def expect(self, t: float) -> Array:
        return jnp.array([jnp.cos(self.w * t)])

    def loss_expect(self, expect: Array) -> Array:
        return expect.real

    def grads_expect(self, t: float) -> PyTree:
        grad_w = -t * jnp.sin(self.w * t)
        return self.Params([grad_w])

    def hess_expect(self, t: float) -> PyTree:
        """Exact second derivative of each expectation-value loss wrt parameters."""
        hess_w = -(t**2) * jnp.cos(self.w * t)
        return self.Params([hess_w])

    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        params: PyTree | None = None,
        t0: ScalarLike | None = None,
    ) -> Result:
        params = self.params_default if params is None else params
        return dq.sesolve(
            self.H(params),
            self.y0(params),
            self.tsave,
            exp_ops=self.Es(params),
            method=method,
            gradient=gradient,
            t0=t0,
        )


# t_end chosen away from a node of sin/cos so neither the gradient nor the
# Hessian is accidentally zero
tsave = np.linspace(0.0, 1.0, 11)
hqubit = HQubit(w=1.3, tsave=tsave)
