import torch
from methodtools import lru_cache
from torch import Tensor

from ..solvers.propagator import Propagator


class SEPropagator(Propagator):
    @lru_cache(maxsize=1)
    def propagator(self, H: Tensor, delta_t: float) -> Tensor:
        # -> (b_H, 1, n, n)
        return torch.matrix_exp(-1j * H * delta_t)

    def forward(self, t: float, delta_t: float, psi: Tensor) -> Tensor:
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        H = self.H(t)
        return self.propagator(H, delta_t) @ psi
