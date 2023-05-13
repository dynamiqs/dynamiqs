import torch

from ..solvers.propagator import Propagator


class SEPropagator(Propagator):
    def forward(self, t, dt, psi):
        return torch.matrix_exp(-1j * self.H(t) * dt) @ psi
