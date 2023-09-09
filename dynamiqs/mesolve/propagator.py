import torch
from torch import Tensor

from ..solvers.propagator import Propagator
from ..solvers.utils import cache
from ..utils.vectorization import operator_to_vector, spost, spre, vector_to_operator
from .me_solver import MESolver


class MEPropagator(MESolver, Propagator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        liouvillian_H = cache(lambda H: -1j * (spre(H) - spost(H)))
        LdagL = self.L.mH @ self.L
        liouvillian_L = (
            spre(self.L) @ spost(self.L.mH) - 0.5 * spre(LdagL) - 0.5 * spost(LdagL)
        ).sum(-3)
        self.liouvillian = cache(lambda H: liouvillian_H(H) + liouvillian_L)

    @cache
    def propagator(self, H: Tensor, delta_t: float) -> Tensor:
        # -> (b_H, 1, n * n, n * n)
        return torch.matrix_exp(self.liouvillian(H) * delta_t)

    def forward(self, t: float, delta_t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        H = self.H(t)
        rho_vec = operator_to_vector(rho)  # (b_H, b_rho, n * n, 1)
        new_rho_vec = self.propagator(H, delta_t) @ rho_vec  # (b_H, b_rho, n * n, 1)
        return vector_to_operator(new_rho_vec)
