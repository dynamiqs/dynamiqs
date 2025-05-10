from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PyTree

import dynamiqs as dq
from dynamiqs import QArray, asqarray, stack
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.options import Options
from dynamiqs.qarrays.layout import Layout
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray


class System(ABC):
    def __init__(self):
        self.n = None
        self.tsave = None
        self.params_default = None

    @abstractmethod
    def H(self, params: PyTree) -> QArray | TimeQArray:
        """Compute the Hamiltonian."""

    @abstractmethod
    def y0(self, params: PyTree) -> QArray:
        """Compute the initial state."""

    @abstractmethod
    def Es(self, params: PyTree) -> list[QArray]:
        """Compute the expectation value operators."""

    def state(self, t: float) -> QArray:
        """Compute the exact state at a given time."""
        raise NotImplementedError

    def states(self, t: Array) -> QArray:
        return stack([self.state(t_.item()) for t_ in t])

    def expect(self, t: float) -> Array:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Array) -> Array:
        return jnp.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def loss_state(self, state: QArray) -> Array:
        """Compute an example loss function from a given state."""
        raise NotImplementedError

    def grads_states(self, t: float) -> PyTree:
        """Compute the exact gradients of the example state loss function with respect
        to the system parameters.
        """
        raise NotImplementedError

    def loss_expect(self, expect: Array) -> Array:
        """Compute example loss functions for each expectation values."""
        return expect.real

    def grads_expect(self, t: float) -> PyTree:
        """Compute the exact gradients of the example expectation values loss functions
        with respect to the system parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        options: Options = Options(),  # noqa: B008
        params: PyTree | None = None,
    ) -> Result:
        pass


class OpenSystem(System):
    @abstractmethod
    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:
        """Compute the jump operators."""


class OCavity(OpenSystem):
    class Params(NamedTuple):
        delta: float
        alpha0: float
        kappa: float

    def __init__(
        self,
        *,
        n: int,
        delta: float,
        alpha0: float,
        kappa: float,
        tsave: ArrayLike,
        layout: Layout,
    ):
        self.n = n
        self.delta = delta
        self.alpha0 = alpha0
        self.kappa = kappa
        self.tsave = tsave
        self.layout = layout

        # define default gradient parameters
        self.params_default = self.Params(delta, alpha0, kappa)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        return params.delta * dq.number(self.n, layout=self.layout)

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:
        return [jnp.sqrt(params.kappa) * dq.destroy(self.n, layout=self.layout)]

    def y0(self, params: PyTree) -> QArray:
        return dq.coherent(self.n, params.alpha0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [
            dq.position(self.n, layout=self.layout),
            dq.momentum(self.n, layout=self.layout),
        ]

    def _alpha(self, t: float) -> Array:
        return self.alpha0 * jnp.exp(-1j * self.delta * t - 0.5 * self.kappa * t)

    def state(self, t: float) -> QArray:
        return dq.coherent_dm(self.n, self._alpha(t))

    def steady_state(self) -> QArray:
        return dq.coherent_dm(self.n, 0.0)

    def expect(self, t: float) -> Array:
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return jnp.array([exp_x, exp_p], dtype=alpha_t.dtype)

    def loss_state(self, state: QArray) -> Array:
        return dq.expect(dq.number(self.n, layout=self.layout), state).real

    def grads_state(self, t: float) -> PyTree:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0 * jnp.exp(-self.kappa * t)
        grad_kappa = -(self.alpha0**2) * t * jnp.exp(-self.kappa * t)
        return self.Params(grad_delta, grad_alpha0, grad_kappa)

    def grads_expect(self, t: float) -> PyTree:
        cdt = jnp.cos(self.delta * t)
        sdt = jnp.sin(self.delta * t)
        emkt = jnp.exp(-0.5 * self.kappa * t)

        grad_x_delta = -self.alpha0 * t * sdt * emkt
        grad_p_delta = -self.alpha0 * t * cdt * emkt
        grad_x_alpha0 = cdt * emkt
        grad_p_alpha0 = -sdt * emkt
        grad_x_kappa = -0.5 * self.alpha0 * t * cdt * emkt
        grad_p_kappa = 0.5 * self.alpha0 * t * sdt * emkt

        return self.Params(
            [grad_x_delta, grad_p_delta],
            [grad_x_alpha0, grad_p_alpha0],
            [grad_x_kappa, grad_p_kappa],
        )


class OTDQubit(OpenSystem):
    class Params(NamedTuple):
        eps: float
        omega: float
        gamma: float

    def __init__(self, *, eps: float, omega: float, gamma: float, tsave: ArrayLike):
        self.n = 2
        self.eps = eps
        self.omega = omega
        self.gamma = gamma
        self.tsave = tsave

        # define default gradient parameters
        self.params_default = self.Params(eps, omega, gamma)

    def H(self, params: PyTree) -> QArray | TimeQArray:
        f = lambda t: params.eps * jnp.cos(params.omega * t) * dq.sigmax()
        return dq.timecallable(f)

    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:
        return [jnp.sqrt(params.gamma) * dq.sigmax()]

    def y0(self, params: PyTree) -> QArray:  # noqa: ARG002
        return dq.fock(2, 0)

    def Es(self, params: PyTree) -> list[QArray]:  # noqa: ARG002
        return [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

    def _theta(self, t: float) -> float:
        return 2 * self.eps / self.omega * jnp.sin(self.omega * t)

    def _eta(self, t: float) -> float:
        return jnp.exp(-2 * self.gamma * t)

    def state(self, t: float) -> QArray:
        theta = self._theta(t)
        eta = self._eta(t)
        rho_00 = 0.5 * (1 + eta * jnp.cos(theta))
        rho_11 = 0.5 * (1 - eta * jnp.cos(theta))
        rho_01 = 0.5j * eta * jnp.sin(theta)
        rho_10 = -0.5j * eta * jnp.sin(theta)
        return asqarray([[rho_00, rho_01], [rho_10, rho_11]])

    def expect(self, t: float) -> Array:
        theta = self._theta(t)
        eta = self._eta(t)
        exp_x = 0
        exp_y = -eta * jnp.sin(theta)
        exp_z = eta * jnp.cos(theta)
        return jnp.array([exp_x, exp_y, exp_z]).real

    def loss_state(self, state: QArray) -> Array:
        return dq.expect(dq.sigmaz(), state).real

    def grads_state(self, t: float) -> PyTree:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_deps = 2 * jnp.sin(self.omega * t) / self.omega
        dtheta_domega = 2 * self.eps / self.omega * t * jnp.cos(self.omega * t)
        dtheta_domega -= 2 * self.eps / self.omega**2 * jnp.sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_eps = -dtheta_deps * eta * jnp.sin(theta)
        grad_omega = -dtheta_domega * eta * jnp.sin(theta)
        grad_gamma = deta_dgamma * jnp.cos(theta)
        return self.Params(grad_eps, grad_omega, grad_gamma)

    def grads_expect(self, t: float) -> PyTree:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_deps = 2 * jnp.sin(self.omega * t) / self.omega
        dtheta_domega = 2 * self.eps / self.omega * t * jnp.cos(self.omega * t)
        dtheta_domega -= 2 * self.eps / self.omega**2 * jnp.sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_z_eps = -dtheta_deps * eta * jnp.sin(theta)
        grad_z_omega = -dtheta_domega * eta * jnp.sin(theta)
        grad_z_gamma = deta_dgamma * jnp.cos(theta)
        # gradients of sigma_y
        grad_y_eps = -dtheta_deps * eta * jnp.cos(theta)
        grad_y_omega = -dtheta_domega * eta * jnp.cos(theta)
        grad_y_gamma = -deta_dgamma * jnp.sin(theta)
        # gradients of sigma_x
        grad_x_eps = 0
        grad_x_omega = 0
        grad_x_gamma = 0
        return self.Params(
            [grad_x_eps, grad_y_eps, grad_z_eps],
            [grad_x_omega, grad_y_omega, grad_z_omega],
            [grad_x_gamma, grad_y_gamma, grad_z_gamma],
        )
