from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import Any

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver
from dynamiqs.solvers.result import Result
from dynamiqs.utils.tensor_types import ArrayLike

from ..system import System


class ClosedSystem(System):
    @property
    def _state_shape(self) -> tuple[int, int]:
        return self.n, 1

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
        return dq.sesolve(
            H,
            y0,
            tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class Cavity(ClosedSystem):
    # `H_batched: (3, n, n)
    # `y0_batched`: (4, n, n)
    # `exp_ops`: (2, n, n)

    def __init__(
        self,
        *,
        n: int,
        delta: float,
        alpha0: float,
        t_end: float,
        requires_grad: bool = False,
    ):
        # store parameters
        self.n = n
        self.delta = torch.as_tensor(delta).requires_grad_(requires_grad)
        self.alpha0 = torch.as_tensor(alpha0).requires_grad_(requires_grad)
        self.t_end = torch.as_tensor(t_end)

        # define gradient parameters
        self.params = (self.delta, self.alpha0)

        # bosonic operators
        a = dq.destroy(self.n)
        adag = a.mH

        # loss operator
        self.loss_op = adag @ a

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.exp_ops = [(a + adag) / sqrt(2), 1j * (adag - a) / sqrt(2)]

        # prepare initial states
        self.y0 = dq.coherent(self.n, self.alpha0)
        self.y0_batched = [
            dq.coherent(self.n, self.alpha0),
            dq.coherent(self.n, 1j * self.alpha0),
            dq.coherent(self.n, -self.alpha0),
            dq.coherent(self.n, -1j * self.alpha0),
        ]

    def tsave(self, num_tsave: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), num_tsave)

    def _alpha(self, t: float) -> Tensor:
        return self.alpha0 * torch.exp(-1j * self.delta * t)

    def state(self, t: float) -> Tensor:
        return dq.coherent(self.n, self._alpha(t))

    def expect(self, t: float) -> Tensor:
        alpha_t = self._alpha(t)
        exp_x = sqrt(2) * alpha_t.real
        exp_p = sqrt(2) * alpha_t.imag
        return torch.tensor([exp_x, exp_p], dtype=alpha_t.dtype)

    def grads_state(self, t: float) -> Tensor:
        grad_delta = 0.0
        grad_alpha0 = 2 * self.alpha0
        return torch.tensor([grad_delta, grad_alpha0]).detach()

    def grads_expect(self, t: float) -> Tensor:
        grad_x_delta = sqrt(2) * self.alpha0 * -t * sin(-self.delta * t)
        grad_p_delta = sqrt(2) * self.alpha0 * -t * cos(-self.delta * t)
        grad_x_alpha0 = sqrt(2) * cos(-self.delta * t)
        grad_p_alpha0 = sqrt(2) * sin(-self.delta * t)

        return torch.tensor([
            [grad_x_delta, grad_x_alpha0],
            [grad_p_delta, grad_p_alpha0],
        ]).detach()


class TDQubit(ClosedSystem):
    def __init__(
        self, *, Omega: float, omega: float, t_end: float, requires_grad: bool = False
    ):
        # store parameters
        self.Omega = torch.as_tensor(Omega).requires_grad_(requires_grad)
        self.omega = torch.as_tensor(omega).requires_grad_(requires_grad)
        self.t_end = torch.as_tensor(t_end)

        # define gradient parameters
        self.parameters = (self.Omega, self.omega)

        # loss operator
        self.loss_op = dq.sigmaz()

        # prepare quantum operators
        self.exp_ops = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]

        # prepare initial states
        self.y0 = dq.fock(2, 0)

    def H(self, t: float) -> Tensor:
        return self.Omega * torch.cos(self.omega * t) * dq.sigmax()

    def tsave(self, num_tsave: int) -> Tensor:
        return torch.linspace(0.0, self.t_end.item(), num_tsave)

    def _theta(self, t: float) -> float:
        return self.Omega / self.omega * sin(self.omega * t)

    def state(self, t: float) -> Tensor:
        theta = self._theta(t)
        return cos(theta) * dq.fock(2, 0) - 1j * sin(theta) * dq.fock(2, 1)

    def expect(self, t: float) -> Tensor:
        theta = self._theta(t)
        return torch.tensor(
            [0, -sin(2 * theta), cos(2 * theta)], dtype=theta.dtype.to_complex()
        )

    def grads_state(self, t: float) -> Tensor:
        theta = self._theta(t)
        # gradients of theta
        dtheta_dOmega = sin(self.omega * t) / self.omega
        dtheta_domega = self.Omega * t * cos(
            self.omega * t
        ) / self.omega - self.Omega / self.omega**2 * sin(self.omega * t)
        # gradients of sigma_z
        grad_Omega = -2 * dtheta_dOmega * sin(2 * theta)
        grad_omega = -2 * dtheta_domega * sin(2 * theta)
        return torch.tensor([grad_Omega, grad_omega]).detach()

    def grads_expect(self, t: float) -> Tensor:
        theta = self._theta(t)
        # gradients of theta
        dtheta_dOmega = sin(self.omega * t) / self.omega
        dtheta_domega = self.Omega * t * cos(
            self.omega * t
        ) / self.omega - self.Omega / self.omega**2 * sin(self.omega * t)
        # gradients of sigma_z
        grad_z_Omega = -2 * dtheta_dOmega * sin(2 * theta)
        grad_z_omega = -2 * dtheta_domega * sin(2 * theta)
        # gradients of sigma_y
        grad_y_Omega = -2 * dtheta_dOmega * cos(2 * theta)
        grad_y_omega = -2 * dtheta_domega * cos(2 * theta)
        # gradients of sigma_x
        grad_x_Omega = 0
        grad_x_omega = 0
        return torch.tensor(
            [
                [grad_x_Omega, grad_x_omega],
                [grad_y_Omega, grad_y_omega],
                [grad_z_Omega, grad_z_omega],
            ]
        ).detach()


cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0, t_end=1.0)
grad_cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0, t_end=1.0, requires_grad=True)

tdqubit = TDQubit(Omega=3.0, omega=10.0, t_end=1.0)
grad_tdqubit = TDQubit(Omega=3.0, omega=10.0, t_end=1.0, requires_grad=True)
