from __future__ import annotations

from math import cos, exp, pi, sin, sqrt
from typing import Any

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver
from dynamiqs.solvers.result import Result
from dynamiqs.utils.tensor_types import ArrayLike
from dynamiqs.utils.tensor_types import dtype_real_to_complex as to_complex

from ..system import System


class OpenSystem(System):
    def __init__(self):
        super().__init__()
        self.jump_ops = None

    @property
    def _state_shape(self) -> tuple[int, int]:
        return self.n, self.n

    def _run(
        self,
        H: Tensor,
        y0: Tensor,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
    ) -> Result:
        return dq.mesolve(
            H,
            self.jump_ops,
            y0,
            tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class LeakyCavity(OpenSystem):
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
        self.exp_ops = [(a + adag) / sqrt(2), 1j * (adag - a) / sqrt(2)]

        # prepare initial states
        self.y0 = dq.coherent_dm(self.n, self.alpha0)
        self.y0_batched = [
            dq.coherent_dm(self.n, self.alpha0),
            dq.coherent_dm(self.n, 1j * self.alpha0),
            dq.coherent_dm(self.n, -self.alpha0),
            dq.coherent_dm(self.n, -1j * self.alpha0),
        ]

    def tsave(self, num_tsave: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), num_tsave)

    def _alpha(self, t: float) -> Tensor:
        return (
            self.alpha0
            * torch.exp(-1j * self.delta * t)
            * torch.exp(-0.5 * self.kappa * t)
        )

    def state(self, t: float) -> Tensor:
        return dq.coherent_dm(self.n, self._alpha(t))

    def expect(self, t: float) -> Tensor:
        alpha_t = self._alpha(t)
        exp_x = sqrt(2) * alpha_t.real
        exp_p = sqrt(2) * alpha_t.imag
        return torch.tensor([exp_x, exp_p], dtype=alpha_t.dtype)

    def grads_state(self, t: float) -> Tensor:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0 * exp(-self.kappa * t)
        grad_kappa = self.alpha0**2 * -t * exp(-self.kappa * t)
        return torch.tensor([grad_delta, grad_alpha0, grad_kappa]).detach()

    # flake8: noqa: E501 line too long
    def grads_expect(self, t: float) -> Tensor:
        # fmt: off
        grad_x_delta = sqrt(2) * self.alpha0 * -t * sin(-self.delta * t) * exp(-0.5 * self.kappa * t)
        grad_p_delta = sqrt(2) * self.alpha0 * -t * cos(-self.delta * t) * exp(-0.5 * self.kappa * t)
        grad_x_alpha0 = sqrt(2) * cos(-self.delta * t) * exp(-0.5 * self.kappa * t)
        grad_p_alpha0 = sqrt(2) * sin(-self.delta * t) * exp(-0.5 * self.kappa * t)
        grad_x_kappa = sqrt(2) * self.alpha0 * cos(-self.delta * t) * -0.5 * t * exp(-0.5 * self.kappa * t)
        grad_p_kappa = sqrt(2) * self.alpha0 * sin(-self.delta * t) * -0.5 * t * exp(-0.5 * self.kappa * t)
        # fmt: on

        return torch.tensor([
            [grad_x_delta, grad_x_alpha0, grad_x_kappa],
            [grad_p_delta, grad_p_alpha0, grad_p_kappa],
        ]).detach()


class DampedTDQubit(OpenSystem):
    def __init__(
        self,
        *,
        Omega: float,
        omega: float,
        gamma: float,
        t_end: float,
        requires_grad: bool = False,
    ):
        # store parameters
        self.Omega = torch.as_tensor(Omega).requires_grad_(requires_grad)
        self.omega = torch.as_tensor(omega).requires_grad_(requires_grad)
        self.gamma = torch.as_tensor(gamma).requires_grad_(requires_grad)
        self.t_end = torch.as_tensor(t_end)

        # define gradient parameters
        self.params = (self.Omega, self.omega, self.gamma)

        # loss operator
        self.loss_op = dq.sigmaz()

        # prepare quantum operators
        self.jump_ops = [torch.sqrt(self.gamma) * dq.sigmax()]
        self.exp_ops = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

        # prepare initial states
        self.y0 = dq.fock(2, 0)

    def H(self, t: float) -> Tensor:
        return self.Omega * torch.cos(self.omega * t) * dq.sigmax()

    def tsave(self, num_tsave: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), num_tsave)

    def _theta(self, t: float) -> float:
        return self.Omega / self.omega * sin(self.omega * t)

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
        return torch.tensor(
            [0, -eta * sin(2 * theta), eta * cos(2 * theta)],
            dtype=to_complex(theta.dtype),
        )

    def grads_state(self, t: float) -> Tensor:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_dOmega = sin(self.omega * t) / self.omega
        dtheta_domega = self.Omega * t * cos(
            self.omega * t
        ) / self.omega - self.Omega / self.omega**2 * sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_Omega = -2 * dtheta_dOmega * eta * sin(2 * theta)
        grad_omega = -2 * dtheta_domega * eta * sin(2 * theta)
        grad_gamma = deta_dgamma * cos(2 * theta)
        return torch.tensor([grad_Omega, grad_omega, grad_gamma]).detach()

    def grads_expect(self, t: float) -> Tensor:
        theta = self._theta(t)
        eta = self._eta(t)
        # gradients of theta
        dtheta_dOmega = sin(self.omega * t) / self.omega
        dtheta_domega = self.Omega * t * cos(
            self.omega * t
        ) / self.omega - self.Omega / self.omega**2 * sin(self.omega * t)
        # gradient of eta
        deta_dgamma = -2 * t * eta
        # gradients of sigma_z
        grad_z_Omega = -2 * dtheta_dOmega * eta * sin(2 * theta)
        grad_z_omega = -2 * dtheta_domega * eta * sin(2 * theta)
        grad_z_gamma = deta_dgamma * cos(2 * theta)
        # gradients of sigma_y
        grad_y_Omega = -2 * dtheta_dOmega * eta * cos(2 * theta)
        grad_y_omega = -2 * dtheta_domega * eta * cos(2 * theta)
        grad_y_gamma = -deta_dgamma * sin(2 * theta)
        # gradients of sigma_x
        grad_x_Omega = 0
        grad_x_omega = 0
        grad_x_gamma = 0
        return torch.tensor([
            [grad_x_Omega, grad_x_omega, grad_x_gamma],
            [grad_y_Omega, grad_y_omega, grad_y_gamma],
            [grad_z_Omega, grad_z_omega, grad_z_gamma],
        ]).detach()


leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0, t_end=1.0)
grad_leaky_cavity_8 = LeakyCavity(
    n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0, t_end=1.0, requires_grad=True
)

damped_tdqubit = DampedTDQubit(Omega=3.0, omega=10.0, gamma=1.0, t_end=1.0)
grad_damped_tdqubit = DampedTDQubit(
    Omega=3.0, omega=10.0, gamma=1.0, t_end=1.0, requires_grad=True
)
