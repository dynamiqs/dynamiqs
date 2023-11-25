from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..mesolve.me_solver import MESolver
from ..solvers.result import Result
from ..solvers.utils import cache
from ..solvers.utils.utils import iteraxis
from ..utils.utils import trace


class SMESolver(MESolver):
    def __init__(
        self,
        *args,
        jump_ops: Tensor,
        etas: Tensor,
        generator: torch.Generator,
    ):
        super().__init__(*args, jump_ops=jump_ops)

        # split jump operators between purely dissipative (eta = 0) and
        # monitored (eta != 0)
        mask = etas.squeeze() == 0.0
        self.Lc = jump_ops[mask]  # (nLc, ..., n, n) purely dissipative
        self.Lm = jump_ops[~mask]  # (nLm, ..., n, n) monitored
        self.etas = etas[~mask]  # (nLm, ...)
        self.generator = generator

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (self.Lm.size(0), *batch_sizes)

        # meas_save: (nLm, ..., len(tmeas) - 1)
        if len(self.tmeas) > 0:
            self.meas_save = torch.zeros(
                *self.meas_shape,
                len(self.tmeas) - 1,
                dtype=self.rdtype,
                device=self.device,
            )
            self.meas_save_iter = iteraxis(self.meas_save, axis=-1)
        else:
            self.meas_save = None

        # tensor to hold the sum of measurement results on a time bin
        # self.bin_meas: (nLm, ...)
        self.bin_meas = torch.zeros(self.meas_shape, device=self.device)

    def run(self) -> Result:
        result = super().run()
        result.meas_save = self.meas_save
        result.tmeas = self.tmeas
        return result

    def _save_meas(self):
        if self.tmeas_counter != 0:
            t_bin = self.tmeas[self.tmeas_counter] - self.tmeas[self.tmeas_counter - 1]
            next(self.meas_save_iter)[:] = self.bin_meas / t_bin
            self.bin_meas = torch.zeros_like(self.bin_meas)

    def sample_wiener(self, dt: float) -> Tensor:
        # -> (nLm, ...)
        return torch.normal(
            torch.zeros(self.meas_shape, device=self.device),
            sqrt(dt),
            generator=self.generator,
        ).to(dtype=self.rdtype)

    @cache
    def Lmp(self, rho: Tensor) -> Tensor:
        # rho: (..., n, n) -> (nLm, ..., n, n)
        # Lm @ rho + rho @ Lmdag
        Lm_rho = self.Lm @ rho
        return Lm_rho + Lm_rho.mH

    @cache
    def exp_val(self, Lmp_rho: Tensor) -> Tensor:
        return trace(Lmp_rho).real

    def diff_backaction(self, dw: Tensor, rho: Tensor) -> Tensor:
        # Compute the measurement backaction term of the diffusive SME.
        # $$ \sum_k \sqrt{\eta_k} \mathcal{M}[Lm_k](\rho) dW_k $$
        # where
        # $$ \mathcal{M}[Lm](\rho) = Lm \rho + \rho Lm^\dag -
        # \mathrm{Tr}\left[(Lm + Lm^\dag) \rho\right] \rho $$

        # rho: (..., n, n) -> (..., n, n)

        # Lm @ rho + rho @ Lmdag
        Lmp_rho = self.Lmp(rho)  # (nLm, ..., n, n)
        exp_val = self.exp_val(Lmp_rho)  # (nLm, ...)

        # Lm @ rho + rho @ Lmdag - Tr(Lm @ rho + rho @ Lmdag) rho
        # tmp: (nLm, ..., n, n)
        tmp = Lmp_rho - exp_val[..., None, None] * rho

        # sum sqrt(eta) * dw * [Lm @ rho + rho @ Lmdag - Tr(Lm @ rho + rho @ Lmdag) rho]
        prefactor = self.etas.sqrt() * dw
        return (prefactor[..., None, None] * tmp).sum(0)

    def update_meas(self, dw: Tensor, rho: Tensor) -> Tensor:
        Lmp_rho = self.Lmp(rho)  # (nLm, ..., n, n)
        exp_val = self.exp_val(Lmp_rho)  # (nLm, ...)
        dy = self.dt * self.etas.sqrt() * exp_val + dw
        self.bin_meas += dy
        return dy  # (nLm, ...)
