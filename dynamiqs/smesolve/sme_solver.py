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
        meas_ops: Tensor,
        etas: Tensor,
        generator: torch.Generator,
        t_meas: Tensor,
    ):
        self.V = meas_ops  # (len(V), n, n)
        self.etas = etas  # (len(V))
        self.generator = generator
        self.t_meas = t_meas  # (len(t_meas))

        super().__init__(*args, jump_ops=jump_ops)

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (*batch_sizes, len(self.V))

        # meas_save: (..., len(V), len(t_meas))
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
    def Vp(self, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, len(V), n, n)
        # V @ rho + rho @ Vdag
        V_rho = torch.einsum('bij,...jk->...bik', self.V, rho)
        return V_rho + V_rho.mH

    @cache
    def exp_val(self, Vp_rho: Tensor) -> Tensor:
        return trace(Vp_rho).real

    def diff_backaction(self, dw: Tensor, rho: Tensor) -> Tensor:
        # Compute the measurement backaction term of the diffusive SME.
        # $$ \sum_k \sqrt{\eta_k} \mathcal{M}[V_k](\rho) dW_k $$
        # where
        # $$ \mathcal{M}[V](\rho) = V \rho + \rho V^\dag -
        # \mathrm{Tr}\left[(V + V^\dag) \rho\right] \rho $$

        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # V @ rho + rho @ Vdag
        Vp_rho = self.Vp(rho)  # (..., len(V), n, n)
        exp_val = self.exp_val(Vp_rho)  # (..., len(V))

        # V @ rho + rho @ Vdag - Tr(V @ rho + rho @ Vdag) rho
        # tmp: (..., len(V), n, n)
        tmp = Vp_rho - exp_val[..., None, None] * rho[..., None, :, :]

        # sum sqrt(eta) * dw * [V @ rho + rho @ Vdag - Tr(V @ rho + rho @ Vdag) rho]
        prefactor = self.etas.sqrt() * dw
        return (prefactor[..., None, None] * tmp).sum(-3)

    def update_meas(self, dw: Tensor, rho: Tensor) -> Tensor:
        Vp_rho = self.Vp(rho)  # (..., len(V), n, n)
        exp_val = self.exp_val(Vp_rho)  # (..., len(V))
        dy = self.etas.sqrt() * exp_val * self.dt + dw  # (..., len(V))
        self.bin_meas += dy
        return dy
