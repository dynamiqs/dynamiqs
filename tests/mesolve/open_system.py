from __future__ import annotations

from math import cos, exp, pi, sin
from typing import Any
from jax import numpy as jnp, Array

import dynamiqs as dq
from dynamiqs import TimeTensor, dag
from dynamiqs.gradient import Gradient
from dynamiqs.solvers import Solver
from dynamiqs.result import Result
from dynamiqs.utils.tensor_types import ArrayLike, dtype_real_to_complex

from ..system import System


class OpenSystem(System):
    def __init__(self):
        super().__init__()
        self.L = None
        self.Lb = None

    def run(
        self,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
        H: ArrayLike | TimeTensor | None = None,
        L: list[ArrayLike] | None = None,
        y0: ArrayLike | None = None,
    ) -> Result:
        H = self.H if H is None else H
        L = self.L if L is None else L
        y0 = self.y0 if y0 is None else y0
        return dq.mesolve(
            H,
            L,
            y0,
            tsave,
            exp_ops=self.E,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class OCavity(OpenSystem):
    # `Hb: (3, n, n)
    # `L`: (2, n, n)
    # `Lb`: (2, 5, n, n)
    # `y0b`: (4, n, n)
    # `E`: (2, n, n)

    def __init__(
        self,
        *,
        n: int,
        kappa: float,
        delta: float,
        alpha0: float,
        t_end: float,
        requires_grad: bool = False,
    ):
        # store parameters
        self.n = n
        self.kappa = jnp.asarray(kappa)
        self.delta = jnp.asarray(delta)
        self.alpha0 = jnp.asarray(alpha0)
        self.t_end = jnp.asarray(t_end)

        # define gradient parameters
        self.params = (self.delta, self.alpha0, self.kappa)

        # bosonic operators
        a = dq.destroy(self.n)
        adag = dag(a)

        # loss operator
        self.loss_op = adag @ a

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.Hb = [0.5 * self.H, self.H, 2 * self.H]
        self.L = [jnp.sqrt(self.kappa) * a, dq.eye(self.n)]
        self.Lb = [L * jnp.arange(5).reshape(5, 1, 1) for L in self.L]
        self.E = [dq.position(self.n), dq.momentum(self.n)]

        # prepare initial states
        self.y0 = dq.coherent_dm(self.n, self.alpha0)
        self.y0b = [
            dq.coherent_dm(self.n, self.alpha0),
            dq.coherent_dm(self.n, 1j * self.alpha0),
            dq.coherent_dm(self.n, -self.alpha0),
            dq.coherent_dm(self.n, -1j * self.alpha0),
        ]

    def tsave(self, n: int) -> Array:
        return jnp.linspace(0.0, self.t_end.item(), n)

    def _alpha(self, t: float) -> Array:
        return self.alpha0 * jnp.exp(-1j * self.delta * t - 0.5 * self.kappa * t)

    def state(self, t: float) -> Array:
        return dq.coherent_dm(self.n, self._alpha(t))

    def expect(self, t: float) -> Array:
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return jnp.array([exp_x, exp_p], dtype=alpha_t.dtype)

    def grads_state(self, t: float) -> Array:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0 * exp(-self.kappa * t)
        grad_kappa = -self.alpha0**2 * t * exp(-self.kappa * t)
        return jnp.array([grad_delta, grad_alpha0, grad_kappa])

    def grads_expect(self, t: float) -> Array:
        cdt = cos(self.delta * t)
        sdt = sin(self.delta * t)
        emkt = exp(-0.5 * self.kappa * t)

        grad_x_delta = -self.alpha0 * t * sdt * emkt
        grad_p_delta = -self.alpha0 * t * cdt * emkt
        grad_x_alpha0 = cdt * emkt
        grad_p_alpha0 = -sdt * emkt
        grad_x_kappa = -0.5 * self.alpha0 * t * cdt * emkt
        grad_p_kappa = 0.5 * self.alpha0 * t * sdt * emkt

        return jnp.array([
            [grad_x_delta, grad_x_alpha0, grad_x_kappa],
            [grad_p_delta, grad_p_alpha0, grad_p_kappa],
        ])


class OTDQubit(OpenSystem):
    def __init__(
        self,
        *,
        eps: float,
        omega: float,
        gamma: float,
        t_end: float,
        requires_grad: bool = False,
    ):
        self.n = 2

        # store parameters
        self.eps = jnp.asarray(eps)
        self.omega = jnp.asarray(omega)
        self.gamma = jnp.asarray(gamma)
        self.t_end = jnp.asarray(t_end)

        # define gradient parameters
        self.params = (self.eps, self.omega, self.gamma)

        # loss operator
        self.loss_op = dq.sigmaz()

        # prepare quantum operators
        self.H = dq.totime(lambda t: self.eps * jnp.cos(self.omega * t) * dq.sigmax())
        self.L = [jnp.sqrt(self.gamma) * dq.sigmax()]
        self.E = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

        # prepare initial states
        self.y0 = dq.fock(2, 0)

    def tsave(self, n: int) -> Array:
        return jnp.linspace(0.0, self.t_end.item(), n)

    def _theta(self, t: float) -> float:
        return self.eps / self.omega * sin(self.omega * t)

    def _eta(self, t: float) -> float:
        return exp(-2 * self.gamma * t)

    def state(self, t: float) -> Array:
        theta = self._theta(t)
        eta = self._eta(t)
        rho_00 = 0.5 * (1 + eta * cos(2 * theta))
        rho_11 = 0.5 * (1 - eta * cos(2 * theta))
        rho_01 = 0.5j * eta * sin(2 * theta)
        rho_10 = -0.5j * eta * sin(2 * theta)
        return jnp.array([[rho_00, rho_01], [rho_10, rho_11]])

    def expect(self, t: float) -> Array:
        theta = self._theta(t)
        eta = self._eta(t)
        exp_x = 0
        exp_y = -eta * sin(2 * theta)
        exp_z = eta * cos(2 * theta)
        return jnp.array(
            [exp_x, exp_y, exp_z],
            dtype=dtype_real_to_complex(theta.dtype),
        )

    def grads_state(self, t: float) -> Array:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_deps = sin(self.omega * t) / self.omega
        dtheta_domega = self.eps * t * cos(
            self.omega * t
        ) / self.omega - self.eps / self.omega**2 * sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_eps = -2 * dtheta_deps * eta * sin(2 * theta)
        grad_omega = -2 * dtheta_domega * eta * sin(2 * theta)
        grad_gamma = deta_dgamma * cos(2 * theta)
        return jnp.array([grad_eps, grad_omega, grad_gamma])

    def grads_expect(self, t: float) -> Array:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_deps = sin(self.omega * t) / self.omega
        dtheta_domega = self.eps * t * cos(
            self.omega * t
        ) / self.omega - self.eps / self.omega**2 * sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_z_eps = -2 * dtheta_deps * eta * sin(2 * theta)
        grad_z_omega = -2 * dtheta_domega * eta * sin(2 * theta)
        grad_z_gamma = deta_dgamma * cos(2 * theta)
        # gradients of sigma_y
        grad_y_eps = -2 * dtheta_deps * eta * cos(2 * theta)
        grad_y_omega = -2 * dtheta_domega * eta * cos(2 * theta)
        grad_y_gamma = -deta_dgamma * sin(2 * theta)
        # gradients of sigma_x
        grad_x_eps = 0
        grad_x_omega = 0
        grad_x_gamma = 0
        return jnp.array([
            [grad_x_eps, grad_x_omega, grad_x_gamma],
            [grad_y_eps, grad_y_omega, grad_y_gamma],
            [grad_z_eps, grad_z_omega, grad_z_gamma],
        ])


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * pi
ocavity = OCavity(n=8, kappa=1.0 * Hz, delta=1.0 * Hz, alpha0=0.5, t_end=0.3)
gocavity = OCavity(
    n=8, kappa=1.0 * Hz, delta=1.0 * Hz, alpha0=0.5, t_end=0.3, requires_grad=True
)

otdqubit = OTDQubit(eps=3.0, omega=10.0, gamma=1.0, t_end=1.0)
gotdqubit = OTDQubit(eps=3.0, omega=10.0, gamma=1.0, t_end=1.0, requires_grad=True)
