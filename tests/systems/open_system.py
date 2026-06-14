from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PyTree, ScalarLike

import dynamiqs as dq
from dynamiqs import QArray, asqarray, dense
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.progress_meter import AbstractProgressMeter
from dynamiqs.qarrays.layout import Layout
from dynamiqs.result import Result
from dynamiqs.time_qarray import TimeQArray

from ._system import System


class OpenSystem(System):
    @abstractmethod
    def Ls(self, params: PyTree) -> list[QArray | TimeQArray]:
        """Compute the jump operators."""

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
        vectorized: bool = False,
        assume_hermitian: bool = True,
    ) -> Result:
        params = self.params_default if params is None else params
        H = self.H(params)
        Ls = self.Ls(params)
        y0 = self.y0(params)
        Es = self.Es(params)
        return dq.mesolve(
            H,
            Ls,
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
            vectorized=vectorized,
            assume_hermitian=assume_hermitian,
        )


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
            jnp.array([grad_x_delta, grad_p_delta]),
            jnp.array([grad_x_alpha0, grad_p_alpha0]),
            jnp.array([grad_x_kappa, grad_p_kappa]),
        )

    def hessian_expect(self, t: float) -> PyTree:
        # second derivatives of (<x>, <p>) = alpha0 e^{-kappa t/2} (cos, -sin)(delta t)
        c = jnp.cos(self.delta * t)
        s = jnp.sin(self.delta * t)
        e = jnp.exp(-0.5 * self.kappa * t)
        a0 = self.alpha0

        d2_delta2 = jnp.array([-a0 * e * t**2 * c, a0 * e * t**2 * s])
        d2_delta_alpha0 = jnp.array([-e * t * s, -e * t * c])
        d2_delta_kappa = jnp.array([0.5 * a0 * t**2 * e * s, 0.5 * a0 * t**2 * e * c])
        d2_alpha02 = jnp.array([0.0, 0.0])
        d2_alpha0_kappa = jnp.array([-0.5 * t * e * c, 0.5 * t * e * s])
        d2_kappa2 = jnp.array([0.25 * a0 * t**2 * e * c, -0.25 * a0 * t**2 * e * s])

        return self.Params(
            self.Params(d2_delta2, d2_delta_alpha0, d2_delta_kappa),
            self.Params(d2_delta_alpha0, d2_alpha02, d2_alpha0_kappa),
            self.Params(d2_delta_kappa, d2_alpha0_kappa, d2_kappa2),
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
            jnp.array([grad_x_eps, grad_y_eps, grad_z_eps]),
            jnp.array([grad_x_omega, grad_y_omega, grad_z_omega]),
            jnp.array([grad_x_gamma, grad_y_gamma, grad_z_gamma]),
        )

    def hessian_expect(self, t: float) -> PyTree:
        # second derivatives of (<x>, <y>, <z>) = (0, -eta sin(theta), eta cos(theta))
        # with theta = 2 eps/omega sin(omega t) and eta = exp(-2 gamma t)
        st, ct = jnp.sin(self.omega * t), jnp.cos(self.omega * t)
        theta = 2 * self.eps / self.omega * st
        eta = jnp.exp(-2 * self.gamma * t)
        sth, cth = jnp.sin(theta), jnp.cos(theta)

        # first derivatives (theta depends on eps/omega, eta depends on gamma)
        th_eps = 2 * st / self.omega
        th_omega = 2 * self.eps * (t * ct / self.omega - st / self.omega**2)
        eta_gamma = -2 * t * eta
        # second derivatives
        th_eps_omega = 2 * (t * ct / self.omega - st / self.omega**2)
        th_omega_omega = (
            2
            * self.eps
            * (
                -(t**2) * st / self.omega
                - 2 * t * ct / self.omega**2
                + 2 * st / self.omega**3
            )
        )
        eta_gamma_gamma = 4 * t**2 * eta

        def leaf(theta_p, theta_q, theta_pq, eta_p, eta_q, eta_pq):
            # second derivative of (0, -eta sin(theta), eta cos(theta)) w.r.t. params
            d2y = -(
                eta_pq * sth
                + cth * (eta_p * theta_q + eta_q * theta_p)
                - eta * sth * theta_p * theta_q
                + eta * cth * theta_pq
            )
            d2z = (
                eta_pq * cth
                - sth * (eta_p * theta_q + eta_q * theta_p)
                - eta * cth * theta_p * theta_q
                - eta * sth * theta_pq
            )
            return jnp.array([0.0, d2y, d2z])

        # parameter order: (eps, omega, gamma)
        return self.Params(
            self.Params(
                leaf(th_eps, th_eps, 0.0, 0.0, 0.0, 0.0),
                leaf(th_eps, th_omega, th_eps_omega, 0.0, 0.0, 0.0),
                leaf(th_eps, 0.0, 0.0, 0.0, eta_gamma, 0.0),
            ),
            self.Params(
                leaf(th_omega, th_eps, th_eps_omega, 0.0, 0.0, 0.0),
                leaf(th_omega, th_omega, th_omega_omega, 0.0, 0.0, 0.0),
                leaf(th_omega, 0.0, 0.0, 0.0, eta_gamma, 0.0),
            ),
            self.Params(
                leaf(0.0, th_eps, 0.0, eta_gamma, 0.0, 0.0),
                leaf(0.0, th_omega, 0.0, eta_gamma, 0.0, 0.0),
                leaf(0.0, 0.0, 0.0, eta_gamma, eta_gamma, eta_gamma_gamma),
            ),
        )


# # we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# # gradients
Hz = 2 * jnp.pi
tsave = np.linspace(0.0, 0.3, 11)
dense_ocavity = OCavity(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, tsave=tsave, layout=dense
)
dia_ocavity = OCavity(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, tsave=tsave, layout=dq.dia
)

tsave = np.linspace(0.0, 1.0, 11)
otdqubit = OTDQubit(eps=3.0, omega=10.0, gamma=1.0, tsave=tsave)
