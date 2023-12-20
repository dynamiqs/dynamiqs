from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from .._utils import cache
from ..solvers.ode.fixed_solver import FixedSolver
from ..utils.utils import unit
from .sme_solver import SMESolver


class SMERouchon(SMESolver, FixedSolver):
    pass


class SMERouchon1(SMERouchon):
    @cache
    def Ms(self, Hnh: Tensor) -> tuple(Tensor, Tensor):
        # Kraus operators
        # -> (..., n, n), (nL, ..., n, n)
        M0d = self.I - 1j * self.dt * Hnh  # (..., n, n)

        M1cs = sqrt(self.dt) * self.Lc  # (nLc, ..., n, n)
        # M1ms: (nLm, ..., n, n)
        M1ms = torch.sqrt(self.dt * (1 - self.etas[..., None, None])) * self.Lm
        M1s = torch.cat([M1cs, M1ms])  # (nL, ..., n, n)

        return M0d, M1s

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (..., n, n)
        H = self.H(t)  # (..., n, n)
        Hnh = self.Hnh(H)  # (..., n, n)
        M0d, M1s = self.Ms(Hnh)  # (..., n, n), (nL, ..., n, n)

        # sample Wiener process
        dw = self.sample_wiener(self.dt)  # (nLm, ...)

        # update measured signal
        dy = self.update_meas(dw, rho)  # (nLm, ...)

        # compute M0
        seta_dy = self.etas.sqrt() * dy  # (nLm, ...)
        M0 = M0d + (seta_dy[..., None, None] * self.Lm).sum(0)  # (..., n, n)

        # compute rho(t+dt)
        rho = M0 @ rho @ M0.mH + (M1s @ rho @ M1s.mH).sum(0)  # (..., n, n)

        return unit(rho)
