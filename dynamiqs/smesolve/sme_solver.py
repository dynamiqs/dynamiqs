from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..mesolve.me_solver import MESolver
from ..solvers.utils import cache
from ..utils.utils import trace


class SMESolver(MESolver):
    def __init__(
        self,
        *args,
        jump_ops: Tensor,
        etas: Tensor,
        generator: torch.Generator,
        tmeas: Tensor,
    ):
        self.tmeas = tmeas  # (len(tmeas))
        super().__init__(*args, jump_ops=jump_ops)

        # split jump operators between purely dissipative (eta = 0) and
        # monitored (eta != 0)
        mask = etas == 0.0
        self.Lc = jump_ops[mask]  # (len(Lc), n, n) purely dissipative
        self.Lm = jump_ops[~mask]  # (len(Lm), n, n) monitored
        self.etas = etas[~mask]  # (len(Lm))
        self.generator = generator

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (*batch_sizes, len(self.Lm))

        # meas_save: (..., len(Lm), len(tmeas))
        if len(tmeas) > 0:
            meas_save = torch.zeros(
                *self.meas_shape,
                len(tmeas) - 1,
                dtype=self.rdtype,
                device=self.device,
            )
        else:
            meas_save = None

        self.result.meas_save = meas_save

        # tensor to hold the sum of measurement results on a time bin
        self.bin_meas = torch.zeros(self.meas_shape)  # (..., len(etas))

    def _init_time_logic(self):
        self.tstop = torch.cat((self.tsave, self.tmeas)).unique().sort()[0]
        self.tstop_counter = 0

        self.tsave_mask = torch.isin(self.tstop, self.tsave)
        self.tsave_counter = 0

        self.tmeas_mask = torch.isin(self.tstop, self.tmeas)
        self.tmeas_counter = 0

    def _save(self, y: Tensor):
        super()._save(y)
        if self.tmeas_mask[self.tstop_counter]:
            self._save_meas()
            self.tmeas_counter += 1

    def _save_meas(self):
        if self.tmeas_counter != 0:
            t_bin = self.tmeas[self.tmeas_counter] - self.tmeas[self.tmeas_counter - 1]
            self.result.meas_save[..., self.tmeas_counter - 1] = self.bin_meas / t_bin
            self.bin_meas = torch.zeros_like(self.bin_meas)

    def sample_wiener(self, dt: float) -> Tensor:
        # -> (b_H, b_rho, ntrajs)
        return torch.normal(
            torch.zeros(*self.meas_shape), sqrt(dt), generator=self.generator
        ).to(dtype=self.rdtype)

    @cache
    def Lmp(self, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, len(Lm), n, n)
        # Lm @ rho + rho @ Lmdag
        Lm_rho = torch.einsum('bij,...jk->...bik', self.Lm, rho)
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

        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # Lm @ rho + rho @ Lmdag
        Lmp_rho = self.Lmp(rho)  # (..., len(Lm), n, n)
        exp_val = self.exp_val(Lmp_rho)  # (..., len(Lm))

        # Lm @ rho + rho @ Lmdag - Tr(Lm @ rho + rho @ Lmdag) rho
        # tmp: (..., len(Lm), n, n)
        tmp = Lmp_rho - exp_val[..., None, None] * rho[..., None, :, :]

        # sum sqrt(eta) * dw * [Lm @ rho + rho @ Lmdag - Tr(Lm @ rho + rho @ Lmdag) rho]
        prefactor = self.etas.sqrt() * dw
        return (prefactor[..., None, None] * tmp).sum(-3)

    def update_meas(self, dw: Tensor, rho: Tensor) -> Tensor:
        Lmp_rho = self.Lmp(rho)  # (..., len(Lm), n, n)
        exp_val = self.exp_val(Lmp_rho)  # (..., len(Lm))
        dy = self.etas.sqrt() * exp_val * self.dt + dw  # (..., len(Lm))
        self.bin_meas += dy
        return dy
