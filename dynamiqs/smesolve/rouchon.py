from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver
from ..solvers.utils import cache, kraus_map
from ..utils.utils import unit
from .sme_solver import SMESolver


class SMERouchon(SMESolver, FixedSolver):
    pass


class SMERouchon1(SMERouchon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define cached operators
        # self.M0 (b_H, 1, n, n)
        self.M0_tmp = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)

        # define M1s
        # self.M1s: # (1, len(L), n, n)
        self.M1s = torch.cat(
            [
                sqrt(self.dt) * self.Lc,
                torch.sqrt(self.dt * (1 - self.etas))[..., None, None] * self.Lm,
            ],
        )[None, ...]

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # compute cached operators
        # H, Hnh, M0_tmp: (b_H, 1, ntrajs, n, n)
        H = self.H(t)
        Hnh = self.Hnh(H)
        M0_tmp = self.M0_tmp(Hnh)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)

        # update measured signal
        dy = self.update_meas(dw, rho)

        # update state
        # M0: (b_H, 1, ntrajs, n, n)
        M0 = M0_tmp + ((self.etas.sqrt() * dy)[..., None, None] * self.Lm).sum(-3)
        rho = M0 @ rho @ M0.mH + kraus_map(rho, self.M1s)
        rho = unit(rho)

        return rho
