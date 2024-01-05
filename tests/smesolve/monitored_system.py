from __future__ import annotations

from math import cos, exp, pi, sin
from typing import Any

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs import TimeArray
from dynamiqs.gradient import Gradient
from dynamiqs.result import Result
from dynamiqs.solver import Solver
from dynamiqs.utils.array_types import ArrayLike

from ..mesolve.open_system import OpenSystem


class MonitoredSystem(OpenSystem):
    def __init__(self):
        super().__init__()
        self.etas = None

    def run(
        self,
        tsave: ArrayLike,
        solver: Solver,
        *,
        gradient: Gradient | None = None,
        options: dict[str, Any] | None = None,
        H: ArrayLike | TimeArray | None = None,
        L: list[ArrayLike] | None = None,
        y0: ArrayLike | None = None,
        ntrajs: int = 10,
    ) -> Result:
        H = self.H if H is None else H
        L = self.L if L is None else L
        y0 = self.y0 if y0 is None else y0
        return dq.smesolve(
            H,
            L,
            self.etas,
            y0,
            tsave,
            exp_ops=self.E,
            ntrajs=ntrajs,
            solver=solver,
            gradient=gradient,
            options=options,
        )


class MCavity(MonitoredSystem):
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
        eta: float,
        t_end: float,
        requires_grad: bool = False,
    ):
        # store parameters
        self.n = n
        self.kappa = torch.as_tensor(kappa).requires_grad_(requires_grad)
        self.delta = torch.as_tensor(delta).requires_grad_(requires_grad)
        self.alpha0 = torch.as_tensor(alpha0).requires_grad_(requires_grad)
        self.etas = torch.tensor([eta, 0.0])
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
        self.Hb = [0.5 * self.H, self.H, 2 * self.H]
        self.L = [torch.sqrt(self.kappa) * a, dq.eye(self.n)]
        self.Lb = [L * torch.arange(5).view(5, 1, 1) for L in self.L]
        self.E = [dq.position(self.n), dq.momentum(self.n)]

        # prepare initial states
        self.y0 = dq.coherent_dm(self.n, self.alpha0)
        self.y0b = [
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


# we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# gradients
Hz = 2 * pi
mcavity = MCavity(n=8, kappa=1.0 * Hz, delta=1.0 * Hz, alpha0=0.5, eta=0.8, t_end=0.3)
gmcavity = MCavity(
    n=8,
    kappa=1.0 * Hz,
    delta=1.0 * Hz,
    alpha0=0.5,
    eta=0.8,
    t_end=0.3,
    requires_grad=True,
)
