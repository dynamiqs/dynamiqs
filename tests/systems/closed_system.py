from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PyTree, ScalarLike

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.progress_meter import AbstractProgressMeter
from dynamiqs.qarrays.layout import Layout
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray

from ._system import System


class ClosedSystem(System):
    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        params: PyTree | None = None,
        save_states: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | bool | None = None,
        t0: ScalarLike | None = None,
        save_extra: Callable[[Array], PyTree] | None = None,
    ) -> Result:
        params = self.params_default if params is None else params
        H = self.H(params)
        y0 = self.y0(params)
        Es = self.Es(params)
        return dq.sesolve(
            H,
            y0,
            self.tsave,
            exp_ops=Es,
            method=method,
            gradient=gradient,
            save_states=save_states,
            cartesian_batching=cartesian_batching,
            progress_meter=progress_meter,
            t0=t0,
            save_extra=save_extra,
        )


class Cavity(ClosedSystem):
    class Params(NamedTuple):
        delta: float
        alpha0: float

    def __init__(
        self, *, n: int, delta: float, alpha0: float, tsave: ArrayLike, layout: Layout
    ):
        self.n = n
        self.delta = delta
        self.alpha0 = alpha0
        self.tsave = tsave
        self.layout = layout

        # define default gradient parameters
        self.params_default = self.Params(delta, alpha0)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        return params.delta * dq.number(self.n, layout=self.layout)

    def y0(self, params: PyTree) -> QArray:
        return dq.coherent(self.n, params.alpha0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [
            dq.position(self.n, layout=self.layout),
            dq.momentum(self.n, layout=self.layout),
        ]

    def _alpha(self, params: PyTree, t: float) -> Array:
        return params.alpha0 * jnp.exp(-1j * params.delta * t)

    def _state(self, params: PyTree, t: float) -> QArray:
        # analytical state as a function of the parameters
        return dq.coherent(self.n, self._alpha(params, t))

    def state(self, t: float) -> QArray:
        return self._state(self.params_default, t)

    def _expect(self, params: PyTree, t: float) -> Array:
        # analytical expectation values (<x>, <p>) as a function of the parameters
        alpha_t = self._alpha(params, t)
        return jnp.array([alpha_t.real, alpha_t.imag])

    def expect(self, t: float) -> Array:
        return self._expect(self.params_default, t)

    def loss_state(self, state: QArray) -> Array:
        return dq.expect(dq.number(self.n), state).real

    def grads_state(self, t: float) -> PyTree:
        def _loss_state(params: PyTree) -> Array:
            return self.loss_state(self._state(params, t))

        return jax.grad(_loss_state)(self.params_default)

    def grads_expect(self, t: float) -> PyTree:
        return jax.jacrev(self._expect)(self.params_default, t)

    def hessian_expect(self, t: float) -> PyTree:
        return jax.hessian(self._expect)(self.params_default, t)


class TDQubit(ClosedSystem):
    class Params(NamedTuple):
        eps: float
        omega: float

    def __init__(self, *, eps: float, omega: float, tsave: ArrayLike):
        self.n = 2
        self.eps = eps
        self.omega = omega
        self.tsave = tsave

        # define default gradient parameters
        self.params_default = self.Params(eps, omega)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        f = lambda t: params.eps * jnp.cos(params.omega * t) * dq.sigmax()
        return dq.timecallable(f)

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.fock(2, 0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

    def _theta(self, params: PyTree, t: float) -> float:
        return 2 * params.eps / params.omega * jnp.sin(params.omega * t)

    def _state(self, params: PyTree, t: float) -> QArray:
        # analytical state as a function of the parameters
        theta_2 = (1 / 2) * self._theta(params, t)
        return jnp.cos(theta_2) * dq.fock(2, 0) - 1j * jnp.sin(theta_2) * dq.fock(2, 1)

    def state(self, t: float) -> QArray:
        return self._state(self.params_default, t)

    def _expect(self, params: PyTree, t: float) -> Array:
        # analytical expectation values (<x>, <y>, <z>) as a function of the parameters
        theta = self._theta(params, t)
        return jnp.array([0.0, -jnp.sin(theta), jnp.cos(theta)])

    def expect(self, t: float) -> Array:
        return self._expect(self.params_default, t)

    def loss_state(self, state: QArray) -> Array:
        return dq.expect(dq.sigmaz(), state).real

    def grads_state(self, t: float) -> PyTree:
        def _loss_state(params: PyTree) -> Array:
            return self.loss_state(self._state(params, t))

        return jax.grad(_loss_state)(self.params_default)

    def grads_expect(self, t: float) -> PyTree:
        return jax.jacrev(self._expect)(self.params_default, t)

    def hessian_expect(self, t: float) -> PyTree:
        return jax.hessian(self._expect)(self.params_default, t)


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * jnp.pi
tsave = np.linspace(0.0, 0.3, 11)
dense_cavity = Cavity(n=8, delta=1.0 * Hz, alpha0=0.5, tsave=tsave, layout=dq.dense)
dia_cavity = Cavity(n=8, delta=1.0 * Hz, alpha0=0.5, tsave=tsave, layout=dq.dia)

tsave = np.linspace(0.0, 1.0, 11)
tdqubit = TDQubit(eps=3.0, omega=10.0, tsave=tsave)
