import torch
from torch import Tensor

from .._utils import cache
from ..solvers.propagator import Propagator
from ..utils.vectorization import operator_to_vector, slindbladian, vector_to_operator
from .me_solver import MESolver


class MEPropagator(MESolver, Propagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lindbladian = cache(lambda H: slindbladian(H, self.L))  # (..., n^2, n^2)
        self.y0 = operator_to_vector(self.y0)  # (..., n^2, 1)

    @cache
    def propagator(self, lindbladian: Tensor, delta_t: float) -> Tensor:
        # -> (..., n^2, n^2)
        return torch.matrix_exp(lindbladian * delta_t)

    def forward(self, t: float, delta_t: float, rho_vec: Tensor) -> Tensor:
        # rho_vec: (..., n^2, 1) -> (..., n^2, 1)
        H = self.H(t)
        lindbladian = self.lindbladian(H)
        propagator = self.propagator(lindbladian, delta_t)
        return propagator @ rho_vec

    def save(self, y: Tensor):
        # override `save` method to convert `y` from vector to operator
        # y: (..., n^2, 1)
        y = vector_to_operator(y)  # (..., n, n)
        super().save(y)
