import torch
from torch import Tensor

from ..solvers.propagator import Propagator


class SEPropagator(Propagator):
    def forward(self, t: float, dt: float, psi: Tensor) -> Tensor:
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        return torch.matrix_exp(-1j * self.H(t) * dt) @ psi
