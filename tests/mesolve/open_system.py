from __future__ import annotations

from math import cos, exp, pi, sin
from typing import Any, List

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver
from dynamiqs.solvers.result import Result
from dynamiqs.utils.tensor_types import ArrayLike, dtype_real_to_complex

from ..system import System


class OpenSystem(System):
    def __init__(self):
        super().__init__()
        self.jump_ops = None

    @property
    def _state_shape(self) -> tuple[int, int]:
        return self.n, self.n

    def run(
        self,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
    ) -> Result:
        return self._run(
            self.H,
            self.jump_ops,
            self.y0,
            tsave,
            solver,
            gradient=gradient,
            options=options,
        )

    def _run(
        self,
        H: Tensor,
        jump_ops: List[ArrayLike] | None,
        y0: Tensor,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
    ) -> Result:
        return dq.mesolve(
            H,
            jump_ops,
            y0,
            tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class OCavity(OpenSystem):
    # `H_batched: (3, n, n)
    # `jump_ops`: (2, n, n)
    # `y0_batched`: (4, n, n)
    # `exp_ops`: (2, n, n)

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
        self.kappa = torch.as_tensor(kappa).requires_grad_(requires_grad)
        self.delta = torch.as_tensor(delta).requires_grad_(requires_grad)
        self.alpha0 = torch.as_tensor(alpha0).requires_grad_(requires_grad)
        self.t_end = torch.as_tensor(t_end)

        # define gradient parameters
        self.params = (self.delta, self.alpha0, self.kappa)

        # bosonic operators
        a = dq.destroy(self.n)
        adag = a.mH

        # loss operator
        self.loss_op = adag @ a

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.jump_ops = [torch.sqrt(self.kappa) * a, dq.eye(self.n)]
        self.jump_ops_batched = [
            L.repeat(4, 1, 1) * torch.linspace(1, 4, 4)[:, None, None]
            for L in self.jump_ops
        ]
        self.exp_ops = [dq.position(self.n), dq.momentum(self.n)]

        # prepare initial states
        self.y0 = dq.coherent_dm(self.n, self.alpha0)
        self.y0_batched = [
            dq.coherent_dm(self.n, self.alpha0),
            dq.coherent_dm(self.n, 1j * self.alpha0),
            dq.coherent_dm(self.n, -self.alpha0),
            dq.coherent_dm(self.n, -1j * self.alpha0),
        ]

    def tsave(self, n: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), n)

    def _alpha(self, t: float) -> Tensor:
        return self.alpha0 * torch.exp(-1j * self.delta * t - 0.5 * self.kappa * t)

    def state(self, t: float) -> Tensor:
        return dq.coherent_dm(self.n, self._alpha(t))

    def expect(self, t: float) -> Tensor:
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return torch.tensor([exp_x, exp_p], dtype=alpha_t.dtype)

    def grads_state(self, t: float) -> Tensor:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0 * exp(-self.kappa * t)
        grad_kappa = -self.alpha0**2 * t * exp(-self.kappa * t)
        return torch.tensor([grad_delta, grad_alpha0, grad_kappa]).detach()

    def grads_expect(self, t: float) -> Tensor:
        cdt = cos(self.delta * t)
        sdt = sin(self.delta * t)
        emkt = exp(-0.5 * self.kappa * t)

        grad_x_delta = -self.alpha0 * t * sdt * emkt
        grad_p_delta = -self.alpha0 * t * cdt * emkt
        grad_x_alpha0 = cdt * emkt
        grad_p_alpha0 = -sdt * emkt
        grad_x_kappa = -0.5 * self.alpha0 * t * cdt * emkt
        grad_p_kappa = 0.5 * self.alpha0 * t * sdt * emkt

        return torch.tensor([
            [grad_x_delta, grad_x_alpha0, grad_x_kappa],
            [grad_p_delta, grad_p_alpha0, grad_p_kappa],
        ]).detach()


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
        # store parameters
        self.eps = torch.as_tensor(eps).requires_grad_(requires_grad)
        self.omega = torch.as_tensor(omega).requires_grad_(requires_grad)
        self.gamma = torch.as_tensor(gamma).requires_grad_(requires_grad)
        self.t_end = torch.as_tensor(t_end)

        # define gradient parameters
        self.params = (self.eps, self.omega, self.gamma)

        # loss operator
        self.loss_op = dq.sigmaz()

        # prepare quantum operators
        self.H = lambda t: self.eps * torch.cos(self.omega * t) * dq.sigmax()
        self.jump_ops = [torch.sqrt(self.gamma) * dq.sigmax()]
        self.exp_ops = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

        # prepare initial states
        self.y0 = dq.fock(2, 0)

    def tsave(self, n: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), n)

    def _theta(self, t: float) -> float:
        return self.eps / self.omega * sin(self.omega * t)

    def _eta(self, t: float) -> float:
        return exp(-2 * self.gamma * t)

    def state(self, t: float) -> Tensor:
        theta = self._theta(t)
        eta = self._eta(t)
        rho_00 = 0.5 * (1 + eta * cos(2 * theta))
        rho_11 = 0.5 * (1 - eta * cos(2 * theta))
        rho_01 = 0.5j * eta * sin(2 * theta)
        rho_10 = -0.5j * eta * sin(2 * theta)
        return torch.tensor([[rho_00, rho_01], [rho_10, rho_11]])

    def expect(self, t: float) -> Tensor:
        theta = self._theta(t)
        eta = self._eta(t)
        exp_x = 0
        exp_y = -eta * sin(2 * theta)
        exp_z = eta * cos(2 * theta)
        return torch.tensor(
            [exp_x, exp_y, exp_z],
            dtype=dtype_real_to_complex(theta.dtype),
        )

    def grads_state(self, t: float) -> Tensor:
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
        return torch.tensor([grad_eps, grad_omega, grad_gamma]).detach()

    def grads_expect(self, t: float) -> Tensor:
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
        return torch.tensor([
            [grad_x_eps, grad_x_omega, grad_x_gamma],
            [grad_y_eps, grad_y_omega, grad_y_gamma],
            [grad_z_eps, grad_z_omega, grad_z_gamma],
        ]).detach()


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * pi
ocavity = OCavity(n=8, kappa=1.0 * Hz, delta=1.0 * Hz, alpha0=0.5, t_end=0.3)
gocavity = OCavity(
    n=8, kappa=1.0 * Hz, delta=1.0 * Hz, alpha0=0.5, t_end=0.3, requires_grad=True
)

otdqubit = OTDQubit(eps=3.0, omega=10.0, gamma=1.0, t_end=1.0)
gotdqubit = OTDQubit(eps=3.0, omega=10.0, gamma=1.0, t_end=1.0, requires_grad=True)
