from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PyTree

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.options import Options
from dynamiqs.result import Result
from dynamiqs.solver import Solver
from dynamiqs.time_array import TimeArray

from ..system import System


class ClosedSystem(System):
    def run(
        self,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: Options = Options(),  # noqa: B008
        params: PyTree | None = None,
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
            solver=solver,
            gradient=gradient,
            options=options,
        )


class Cavity(ClosedSystem):
    class Params(NamedTuple):
        delta: float
        alpha0: float

    def __init__(self, *, n: int, delta: float, alpha0: float, tsave: ArrayLike):
        self.n = n
        self.delta = delta
        self.alpha0 = alpha0
        self.tsave = tsave

        # define default gradient parameters
        self.params_default = self.Params(delta, alpha0)

    def H(self, params: PyTree) -> ArrayLike | TimeArray:
        return params.delta * dq.number(self.n)

    def y0(self, params: PyTree) -> Array:
        return dq.coherent(self.n, params.alpha0)

    def Es(self, params: PyTree) -> Array:  # noqa: ARG002
        return jnp.stack([dq.position(self.n), dq.momentum(self.n)])

    def _alpha(self, t: float) -> Array:
        return self.alpha0 * jnp.exp(-1j * self.delta * t)

    def state(self, t: float) -> Array:
        return dq.coherent(self.n, self._alpha(t))

    def expect(self, t: float) -> Array:
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return jnp.array([exp_x, exp_p], dtype=alpha_t.dtype)

    def loss_state(self, state: Array) -> Array:
        return dq.expect(dq.number(self.n), state).real

    def grads_state(self, t: float) -> PyTree:  # noqa: ARG002
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0
        return self.Params(grad_delta, grad_alpha0)

    def grads_expect(self, t: float) -> PyTree:
        cdt = jnp.cos(self.delta * t)
        sdt = jnp.sin(self.delta * t)

        grad_x_delta = -self.alpha0 * t * sdt
        grad_p_delta = -self.alpha0 * t * cdt
        grad_x_alpha0 = cdt
        grad_p_alpha0 = -sdt

        return self.Params([grad_x_delta, grad_p_delta], [grad_x_alpha0, grad_p_alpha0])


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

    def H(self, params: PyTree) -> TimeArray:
        f = lambda t: params.eps * jnp.cos(params.omega * t) * dq.sigmax()
        return dq.timecallable(f)

    def y0(self, params: PyTree) -> Array:  # noqa: ARG002
        return dq.fock(2, 0)

    def Es(self, params: PyTree) -> Array:  # noqa: ARG002
        return jnp.stack([dq.sigmax(), dq.sigmay(), dq.sigmaz()])

    def _theta(self, t: float) -> float:
        return 2 * self.eps / self.omega * jnp.sin(self.omega * t)

    def state(self, t: float) -> Array:
        theta_2 = (1 / 2) * self._theta(t)
        return jnp.cos(theta_2) * dq.fock(2, 0) - 1j * jnp.sin(theta_2) * dq.fock(2, 1)

    def expect(self, t: float) -> Array:
        theta = self._theta(t)
        exp_x = 0
        exp_y = -jnp.sin(theta)
        exp_z = jnp.cos(theta)
        return jnp.array([exp_x, exp_y, exp_z]).real

    def loss_state(self, state: Array) -> Array:
        return dq.expect(dq.sigmaz(), state).real

    def grads_state(self, t: float) -> PyTree:
        theta = self._theta(t)
        # gradients of theta
        dtheta_deps = 2 * jnp.sin(self.omega * t) / self.omega
        dtheta_domega = 2 * self.eps * t * jnp.cos(self.omega * t) / self.omega
        dtheta_domega -= 2 * self.eps / self.omega**2 * jnp.sin(self.omega * t)
        # gradients of sigma_z
        grad_eps = -dtheta_deps * jnp.sin(theta)
        grad_omega = -dtheta_domega * jnp.sin(theta)
        return self.Params(grad_eps, grad_omega)

    def grads_expect(self, t: float) -> PyTree:
        theta = self._theta(t)
        # gradients of theta
        dtheta_deps = 2 * jnp.sin(self.omega * t) / self.omega
        dtheta_domega = 2 * self.eps * t * jnp.cos(self.omega * t) / self.omega
        dtheta_domega -= 2 * self.eps / self.omega**2 * jnp.sin(self.omega * t)
        # gradients of sigma_z
        grad_z_eps = -dtheta_deps * jnp.sin(theta)
        grad_z_omega = -dtheta_domega * jnp.sin(theta)
        # gradients of sigma_y
        grad_y_eps = -dtheta_deps * jnp.cos(theta)
        grad_y_omega = -dtheta_domega * jnp.cos(theta)
        # gradients of sigma_x
        grad_x_eps = 0
        grad_x_omega = 0
        return self.Params(
            [grad_x_eps, grad_y_eps, grad_z_eps],
            [grad_x_omega, grad_y_omega, grad_z_omega],
        )


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * jnp.pi
tsave = np.linspace(0.0, 0.3, 11)
cavity = Cavity(n=8, delta=1.0 * Hz, alpha0=0.5, tsave=tsave)

tsave = np.linspace(0.0, 1.0, 11)
tdqubit = TDQubit(eps=3.0, omega=10.0, tsave=tsave)
