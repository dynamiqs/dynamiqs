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
        t_save_y: Tensor,
        t_save_meas: Tensor,
    ):
        t_save = torch.cat((t_save_y, t_save_meas)).unique().sort()[0]
        super().__init__(H, y0, t_save, t_save_y, exp_ops, options, jump_ops=jump_ops)

        self.meas_ops = meas_ops  # (len(meas_ops), n, n)
        self.t_save_meas = t_save_meas  # (len(t_save_meas))
        self.etas = etas  # (len(meas_ops))
        self.generator = generator

        self.save_meas_counter = 0

        # initialize additional save tensors
        batch_sizes = self.y0.shape[:-2]

        self.meas_shape = (*batch_sizes, len(self.meas_ops))

        # meas_save: (..., len(meas_ops), len(t_save_meas))
        if len(t_save_meas) > 0:
            self.meas_save = torch.zeros(
                *self.meas_shape,
                len(self.t_save_meas) - 1,
                dtype=self.dtype_real,
                device=self.device,
            )
        else:
            self.meas_save = None

        # tensor to hold the sum of measurement results on a time bin
        self.bin_meas = torch.zeros(self.meas_shape)  # (..., len(etas))

    def _save(self, t: float, y: Tensor):
        if t in self.t_save_y:
            super()._save(t, y)
        if t in self.t_save_meas:
            self.save_meas()

    def save_meas(self):
        if self.save_meas_counter != 0:
            t_bin = (
                self.t_save_meas[self.save_meas_counter]
                - self.t_save_meas[self.save_meas_counter - 1]
            )
            self.meas_save[..., self.save_meas_counter - 1] = self.bin_meas / t_bin
            self.bin_meas = torch.zeros_like(self.bin_meas)
        self.save_meas_counter += 1

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

    def diff_backaction(self, rho: Tensor, dw: Tensor) -> Tensor:
        # Compute the measurement backaction term of the diffusive SME.
        # $$ \sum_k \sqrt{\eta_k} \mathcal{M}[L_k](\rho) dW_k $$
        # where
        # $$ \mathcal{M}[L](\rho) = L \rho + \rho L^\dag -
        # \mathrm{Tr}\left[(L + L^\dag) \rho\right] \rho $$

        # rho: (b_H, b_rho, ntrajs, n, n) -> (b_H, b_rho, ntrajs, n, n)

        # L @ rho + rho @ Ldag
        Lp_rho = self.Lp(rho)  # (..., b, n, n)

        # L @ rho + rho @ Ldag - Tr(L @ rho + rho @  Ldag) rho
        tmp = Lp_rho - torch.einsum('...bii,...jk->...bjk', Lp_rho, rho)
        # tmp: (..., b, n, n)

        # sum sqrt(eta) * dw * [L @ rho + rho @ Ldag - Tr(L @ rho + rho @ Ldag) rho]
        fact = (self.etas.sqrt() * dw).to(self.dtype)
        return torch.einsum('...b,...bij->...ij', fact, tmp)

    def update_meas(self, rho: Tensor, dw: Tensor):
        tr = trace(self.Lp(rho)).real
        self.bin_meas += self.etas.sqrt() * tr * self.dt + dw
