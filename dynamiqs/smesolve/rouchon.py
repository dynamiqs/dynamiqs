from __future__ import annotations

import torch
from torch import Tensor

from ..mesolve.rouchon import MERouchon1
from ..solvers.ode.fixed_solver import FixedSolver
from ..solvers.utils import kraus_map
from ..utils.utils import unit
from .sme_solver import SMESolver


class SMERouchon(SMESolver, FixedSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n = self.H.shape[-1]
        self.I = torch.eye(self.n, device=self.device, dtype=self.cdtype)  # (n, n)


class SMERouchon1(SMERouchon, MERouchon1):
    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # compute cached operators
        # H, Hnh, M0: (b_H, 1, ntrajs, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0 = self.M0(Hnh)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)

        # update measured signal
        dy = self.update_meas(dw, rho)

        # update state
        # M, rho: (b_H, b_rho, ntrajs, n, n)
        M = M0 + (self.etas.sqrt() * self.V @ dy).sum(-3)
        rho = (
            kraus_map(rho, M)
            + kraus_map(rho, self.M1s)
            + (1 - self.etas) * kraus_map(rho, self.V) * self.dt
        )

        rho = unit(rho)

        return rho
