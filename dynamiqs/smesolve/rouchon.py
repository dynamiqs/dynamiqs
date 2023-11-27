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

        self.M0_tmp = cache(lambda Hnh: self.I - 1j * self.dt * Hnh)  # (..., n, n)
        self.M1s = torch.cat(
            [
                sqrt(self.dt) * self.Lc,
                torch.sqrt(self.dt * (1 - self.etas))[..., None, None] * self.Lm,
            ],
        )  # (nL, ..., n, n)

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)

        H = self.H(t)  # (..., n, n)
        Hnh = self.Hnh(H)  # (..., n, n)
        M0_tmp = self.M0_tmp(Hnh)  # (..., n, n)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)  # (nLm, ...)

        # update measured signal
        dy = self.update_meas(dw, rho)  # (nLm, ...)

        # compute M0
        seta_dy = self.etas.sqrt() * dy  # (nLm, ...)
        M0 = M0_tmp + (seta_dy[..., None, None] * self.Lm).sum(0)  # (..., n, n)

        # compute rho(t+dt)
        rho = M0 @ rho @ M0.mH + (self.M1s @ rho @ self.M1s.mH).sum(0)  # (..., n, n)

        return unit(rho)
