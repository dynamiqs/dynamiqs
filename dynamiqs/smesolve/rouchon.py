from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver
from ..solvers.utils import cache
from ..utils.utils import unit
from .sme_solver import SMESolver


class SMERouchon(SMESolver, FixedSolver):
    pass


class SMERouchon1(SMERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define cached operators
        # self.M0 (b_H, b_L, 1, 1, n, n)
        self.M0_tmp = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)

        # define M1s
        # self.M1s: # (len(L), 1, b_L, 1, 1, n, n)
        self.M1s = torch.cat(
            [
                sqrt(self.dt) * self.Lc,
                torch.sqrt(self.dt * (1 - self.etas))[..., None, None] * self.Lm,
            ],
        )

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_L, b_rho, ntrajs, n, n) -> (b_H, b_L, b_rho, ntrajs, n, n)

        # compute cached operators
        # H, Hnh, M0_tmp: (b_H, b_L, 1, ntrajs, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0_tmp = self.M0_tmp(Hnh)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)  # (len(Lm), b_H, b_L, b_rho, ntrajs)

        # update measured signal
        dy = self.update_meas(dw, rho)  # (len(Lm), b_H, b_L, b_rho, ntrajs)

        # compute M0
        # M0: (b_H, b_L, 1, ntrajs, n, n)
        seta_dy = self.etas.sqrt() * dy  # (len(Lm), b_H, b_L, b_rho, ntrajs)
        M0 = M0_tmp + (seta_dy[..., None, None] * self.Lm).sum(0)

        # update state
        rho = M0 @ rho @ M0.mH + (self.M1s @ rho @ self.M1s.mH).sum(0)

        return unit(rho)
