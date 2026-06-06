from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.time_qarray import TimeQArray

from .open_system import OpenSystem


class MeSolveHQubit(OpenSystem):
    r"""Closed-system mesolve qubit with an analytically known Hessian."""

    class Params(NamedTuple):
        w: float

    def __init__(self, *, w: float, tsave: Array):
        self.n = 2
        self.w = w
        self.tsave = tsave
        self.params_default = self.Params(w)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        return 0.5 * params.w * dq.sigmax()

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:  # noqa: ARG002
        return []

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
        hess_w = -(t**2) * jnp.cos(self.w * t)
        return self.Params([hess_w])


tsave = np.linspace(0.0, 1.0, 11)
mesolve_hqubit = MeSolveHQubit(w=1.3, tsave=tsave)