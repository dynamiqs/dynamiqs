import torch

from ..propagator import Propagator


class SEPropagator(Propagator):
    def __init__(self, *args):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)

    def forward(self, t, dt, psi):
        return torch.matrix_exp(-1j * self.H * dt) @ psi
