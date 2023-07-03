from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from ..mesolve.me_solver import MESolver
from ..options import Options
from ..utils.solver_utils import cache
from ..utils.td_tensor import TDTensor
from ..utils.utils import trace


class SMESolver(MESolver):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
        exp_ops: Tensor,
        options: Options,
        *,
        jump_ops: Tensor,
        meas_ops: Tensor,
        etas: Tensor,
        generator: torch.Generator,
        t_save: Tensor,
        t_meas: Tensor,
    ):
        t_save_all = torch.cat((t_save, t_meas)).unique().sort()[0]
        super().__init__(H, y0, t_save_all, t_save, exp_ops, options, jump_ops=jump_ops)

        self.meas_ops = meas_ops  # (len(meas_ops), n, n)
        self.t_meas = t_meas  # (len(t_meas))
        self.etas = etas  # (len(meas_ops))
        self.generator = generator

        # save logic for meas
        self.t_meas_mask = torch.isin(self.t_save_all, t_meas)
        self.t_meas_counter = 0

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (*batch_sizes, len(self.meas_ops))

        # meas_save: (..., len(meas_ops), len(t_meas))
        if len(t_meas) > 0:
            self.meas_save = torch.zeros(
                *self.meas_shape,
                len(t_meas) - 1,
                dtype=self.dtype_real,
                device=self.device,
            )
        else:
            self.meas_save = None

        # tensor to hold the sum of measurement results on a time bin
        self.bin_meas = torch.zeros(self.meas_shape)  # (..., len(etas))

    def _save(self, y: Tensor):
        super()._save(y)
        if self.t_meas_mask[self.t_save_all_counter]:
            self._save_meas(y)
            self.t_meas_counter += 1

    def _save_meas(self, y: Tensor):
        if self.t_meas_counter != 0:
            t_bin = (
                self.t_meas[self.t_meas_counter] - self.t_meas[self.t_meas_counter - 1]
            )
            self.meas_save[..., self.t_meas_counter - 1] = self.bin_meas / t_bin
            self.bin_meas = torch.zeros_like(self.bin_meas)

    def sample_wiener(self, dt: float) -> Tensor:
        # -> (b_H, b_rho, ntrajs)
        return torch.normal(
            torch.zeros(*self.meas_shape), sqrt(dt), generator=self.generator
        ).to(dtype=self.dtype_real)

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
