import torch
from torch import Tensor

from ..solvers.propagator import Propagator
from ..solvers.utils import cache


class SEPropagator(Propagator):
    @cache
    def propagator(self, delta_t: float) -> Tensor:
        # -> (..., n, n)
        return torch.matrix_exp(-1j * self.H * delta_t)

    def forward(self, t: float, delta_t: float, psi: Tensor) -> Tensor:
        # psi: (..., n, 1) -> (..., n, 1)
        return self.propagator(delta_t) @ psi
