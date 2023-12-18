import torch
from torch import Tensor

from .._utils import cache
from ..solvers.ode.fixed_solver import FixedSolver


class SEBackwardEuler(FixedSolver):
    def __init__(self, *args):
        super().__init__(*args)
        n = self.H.size(-1)
        self.I = torch.eye(n, device=self.device, dtype=self.cdtype)  # (n, n)

    @cache
    def Minv(self, H: Tensor) -> Tensor:
        return torch.linalg.inv(self.I + 1j * self.dt * H)

    def forward(self, t: float, psi: Tensor) -> Tensor:
        # psi: (b_H, b_psi, n, 1) -> (b_H, b_psi, n, 1)
        H = self.H(t)
        Minv = self.Minv(H)
        return Minv @ psi
