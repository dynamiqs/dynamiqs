from __future__ import annotations

from math import cos, pi, sin
from typing import Any

import numpy as np
from jax import Array
from jax import numpy as jnp

import dynamiqs as dq
from dynamiqs import dag
from dynamiqs.gradient import Gradient
from dynamiqs.result import Result
from dynamiqs.solver import Solver
from dynamiqs.utils.array_types import ArrayLike, dtype_real_to_complex

from ..system import System


class ClosedSystem(System):
    def run(
        self,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
        params: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> Result:
        H = self.H()
        if y0 is None:
            y0 = self.y0
        return dq.sesolve(
            H,
            y0,
            tsave,
            exp_ops=self.E,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class Cavity(ClosedSystem):
    # `Hb: (3, n, n)
    # `y0b`: (4, n, n)
    # `E`: (2, n, n)

    def __init__(
        self,
        *,
        n: int,
        delta: float,
        alpha0: float,
        t_end: float,
    ):
        # store parameters
        self.n = n
        self.delta = jnp.asarray(delta)
        self.alpha0 = jnp.asarray(alpha0)
        self.t_end = jnp.asarray(t_end)

        # define gradient parameters
        self.params = (self.delta, self.alpha0)

        # bosonic operators
        a = dq.destroy(self.n)
        adag = dag(a)

        # loss operator
        self.loss_op = adag @ a

        # prepare quantum operators
        self.E = [dq.position(self.n), dq.momentum(self.n)]

        # prepare initial states
        self.y0 = dq.coherent(self.n, self.alpha0)
        self.y0b = [
            dq.coherent(self.n, self.alpha0),
            dq.coherent(self.n, 1j * self.alpha0),
            dq.coherent(self.n, -self.alpha0),
            dq.coherent(self.n, -1j * self.alpha0),
        ]

    def H(self, params: Array = None):
        delta, alpha0 = params if params is not None else self.params
        return delta * dq.number(self.n)

    def Hb(self, params: Array):
        H = self.H(params)
        return [0.5 * H, H, 2 * H]

    def tsave(self, n: int) -> ArrayLike:
        return np.linspace(0.0, self.t_end.item(), n)

    def _alpha(self, t: float) -> Array:
        return self.alpha0 * jnp.exp(-1j * self.delta * t)

    def state(self, t: float) -> Array:
        return dq.coherent(self.n, self._alpha(t))

    def expect(self, t: float) -> Array:
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return jnp.array([exp_x, exp_p], dtype=alpha_t.dtype)

    def grads_state(self, t: float) -> Array:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0
        return jnp.array([grad_delta, grad_alpha0])

    def grads_expect(self, t: float) -> Array:
        cdt = cos(self.delta * t)
        sdt = sin(self.delta * t)

        grad_x_delta = -self.alpha0 * t * sdt
        grad_p_delta = -self.alpha0 * t * cdt
        grad_x_alpha0 = cdt
        grad_p_alpha0 = -sdt

        return jnp.array([
            [grad_x_delta, grad_x_alpha0],
            [grad_p_delta, grad_p_alpha0],
        ])


class TDQubit(ClosedSystem):
    def __init__(self, *, eps: float, omega: float, t_end: float):
        self.n = 2

        # store parameters
        self.eps = jnp.asarray(eps)
        self.omega = jnp.asarray(omega)
        self.t_end = jnp.asarray(t_end)

        # define gradient parameters
        self.params = (self.eps, self.omega)

        # loss operator
        self.loss_op = dq.sigmaz()

        # prepare quantum operators
        self.E = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

        # prepare initial states
        self.y0 = dq.fock(2, 0)

    def H(self, params: Array = None):
        eps, omega = params if params is not None else self.params
        return dq.totime(lambda t, args: eps * jnp.cos(omega * t) * dq.sigmax())

    def tsave(self, n: int) -> Array:
        return jnp.linspace(0.0, self.t_end.item(), n)

    def _theta(self, t: float) -> float:
        return self.eps / self.omega * sin(self.omega * t)

    def state(self, t: float) -> Array:
        theta = self._theta(t)
        return cos(theta) * dq.fock(2, 0) - 1j * sin(theta) * dq.fock(2, 1)

    def expect(self, t: float) -> Array:
        theta = self._theta(t)
        exp_x = 0
        exp_y = -sin(2 * theta)
        exp_z = cos(2 * theta)
        return jnp.array(
            [exp_x, exp_y, exp_z],
            dtype=dtype_real_to_complex(theta.dtype),
        )

    def grads_state(self, t: float) -> Array:
        theta = self._theta(t)
        # gradients of theta
        dtheta_deps = sin(self.omega * t) / self.omega
        dtheta_domega = self.eps * t * cos(
            self.omega * t
        ) / self.omega - self.eps / self.omega**2 * sin(self.omega * t)
        # gradients of sigma_z
        grad_eps = -2 * dtheta_deps * sin(2 * theta)
        grad_omega = -2 * dtheta_domega * sin(2 * theta)
        return jnp.array([grad_eps, grad_omega])

    def grads_expect(self, t: float) -> Array:
        theta = self._theta(t)
        # gradients of theta
        dtheta_deps = sin(self.omega * t) / self.omega
        dtheta_domega = self.eps * t * cos(
            self.omega * t
        ) / self.omega - self.eps / self.omega**2 * sin(self.omega * t)
        # gradients of sigma_z
        grad_z_eps = -2 * dtheta_deps * sin(2 * theta)
        grad_z_omega = -2 * dtheta_domega * sin(2 * theta)
        # gradients of sigma_y
        grad_y_eps = -2 * dtheta_deps * cos(2 * theta)
        grad_y_omega = -2 * dtheta_domega * cos(2 * theta)
        # gradients of sigma_x
        grad_x_eps = 0
        grad_x_omega = 0
        return jnp.array([
            [grad_x_eps, grad_x_omega],
            [grad_y_eps, grad_y_omega],
            [grad_z_eps, grad_z_omega],
        ])


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * pi
cavity = Cavity(n=8, delta=1.0 * Hz, alpha0=0.5, t_end=0.3)
gcavity = Cavity(n=8, delta=1.0 * Hz, alpha0=0.5, t_end=0.3)

tdqubit = TDQubit(eps=3.0, omega=10.0, t_end=1.0)
gtdqubit = TDQubit(eps=3.0, omega=10.0, t_end=1.0)
