import torch
from torch import Tensor

from ..solvers.propagator import Propagator
from ..solvers.utils import cache
from ..utils.vectorization import operator_to_vector, slindbladian, vector_to_operator
from .me_solver import MESolver


class MEPropagator(MESolver, Propagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.lindbladian: (b_H, 1, n^2, n^2)
        self.lindbladian = cache(lambda H: slindbladian(H, self.L))
        self.y0 = operator_to_vector(self.y0)  # (b_H, b_rho, n^2, 1)

    @cache
    def propagator(self, lindbladian: Tensor, delta_t: float) -> Tensor:
        # -> (b_H, 1, n^2, n^2)
        return torch.matrix_exp(lindbladian * delta_t)

    def forward(self, t: float, delta_t: float, rho_vec: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n^2, 1) -> (b_H, b_rho, n^2, 1)
        H = self.H(t)
        lindbladian = self.lindbladian(H)
        return self.propagator(lindbladian, delta_t) @ rho_vec  # (b_H, b_rho, n^2, 1)

    def save(self, y: Tensor):
        # override `save` method to convert `y` from vector to operator
        y = vector_to_operator(y)  # (b_H, b_rho, n, n)
        super().save(y)
