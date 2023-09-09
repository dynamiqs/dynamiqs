import torch
from torch import Tensor

from ..solvers.propagator import Propagator
from ..solvers.utils import cache
from ..utils.vectorization import operator_to_vector, slindbladian, vector_to_operator
from .me_solver import MESolver


class MEPropagator(MESolver, Propagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lindbladian = slindbladian(self.H, self.L)

    @cache
    def propagator(self, delta_t: float) -> Tensor:
        # -> (b_H, 1, n * n, n * n)
        return torch.matrix_exp(self.lindbladian * delta_t)

    def forward(self, t: float, delta_t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        rho_vec = operator_to_vector(rho)  # (b_H, b_rho, n * n, 1)
        new_rho_vec = self.propagator(delta_t) @ rho_vec  # (b_H, b_rho, n * n, 1)
        return vector_to_operator(new_rho_vec)
