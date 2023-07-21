from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..mesolve.me_solver import MESolver
from ..solvers.utils.utils import cache
from ..utils.utils import trace


class SMESolver(MESolver):
    def __init__(
        self,
        *args,
        jump_ops: Tensor,
        meas_ops: Tensor,
        etas: Tensor,
        generator: torch.Generator,
        t_meas: Tensor,
    ):
        self.meas_ops = meas_ops  # (len(meas_ops), n, n)
        self.etas = etas  # (len(meas_ops))
        self.generator = generator
        self.t_meas = t_meas  # (len(t_meas))

        super().__init__(*args, jump_ops=jump_ops)

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (*batch_sizes, len(self.meas_ops))

        # meas_save: (..., len(meas_ops), len(t_meas))
        if len(t_meas) > 0:
            meas_save = torch.zeros(
                *self.meas_shape,
                len(t_meas) - 1,
                dtype=self.rdtype,
                device=self.device,
            )
        else:
            meas_save = None

        self.result.meas_save = meas_save

        # tensor to hold the sum of measurement results on a time bin
        self.bin_meas = torch.zeros(self.meas_shape)  # (..., len(etas))

    def _init_time_logic(self):
        self.t_stop = torch.cat((self.t_save, self.t_meas)).unique().sort()[0]
        self.t_stop_counter = 0

        self.t_save_mask = torch.isin(self.t_stop, self.t_save)
        self.t_save_counter = 0

        self.t_meas_mask = torch.isin(self.t_stop, self.t_meas)
        self.t_meas_counter = 0

    def _save(self, y: Tensor):
        super()._save(y)
        if self.t_meas_mask[self.t_stop_counter]:
            self._save_meas()
            self.t_meas_counter += 1

    def _save_meas(self):
        if self.t_meas_counter != 0:
            t_bin = (
                self.t_meas[self.t_meas_counter] - self.t_meas[self.t_meas_counter - 1]
            )
            self.result.meas_save[..., self.t_meas_counter - 1] = self.bin_meas / t_bin
            self.bin_meas = torch.zeros_like(self.bin_meas)

    def sample_wiener(self, dt: float) -> Tensor:
        # -> (b_H, b_rho, ntrajs)
        return torch.normal(
            torch.zeros(*self.meas_shape), sqrt(dt), generator=self.generator
        ).to(dtype=self.rdtype)

    @cache
    def Lp(self, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)
        # L @ rho + rho @ Ldag
        L_rho = torch.einsum('bij,...jk->...bik', self.meas_ops, rho)
        return L_rho + L_rho.mH

    @cache
    def exp_val(self, Lp_rho: Tensor) -> Tensor:
        return trace(Lp_rho).real

    def diff_backaction(self, dw: Tensor, rho: Tensor) -> Tensor:
        # Compute the measurement backaction term of the diffusive SME.
        # $$ \sum_k \sqrt{\eta_k} \mathcal{M}[L_k](\rho) dW_k $$
        # where
        # $$ \mathcal{M}[L](\rho) = L \rho + \rho L^\dag -
        # \mathrm{Tr}\left[(L + L^\dag) \rho\right] \rho $$

        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # L @ rho + rho @ Ldag
        Lp_rho = self.Lp(rho)  # (..., b, n, n)
        exp_val = self.exp_val(Lp_rho)  # (..., b)

        # L @ rho + rho @ Ldag - Tr(L @ rho + rho @  Ldag) rho
        tmp = Lp_rho - exp_val[..., None, None] * rho[..., None, :, :]  # (..., b, n, n)

        # sum sqrt(eta) * dw * [L @ rho + rho @ Ldag - Tr(L @ rho + rho @ Ldag) rho]
        prefactor = self.etas.sqrt() * dw
        return (prefactor[..., None, None] * tmp).sum(-3)

    def update_meas(self, dw: Tensor, rho: Tensor):
        Lp_rho = self.Lp(rho)  # (..., b, n, n)
        exp_val = self.exp_val(Lp_rho)  # (..., b)
        self.bin_meas += self.etas.sqrt() * exp_val * self.dt + dw
