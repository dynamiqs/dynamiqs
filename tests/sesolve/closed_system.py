from abc import ABC, abstractmethod
from math import pi, sqrt

import torch
from torch import Tensor

import torchqdynamics as tq


class ClosedSystem(ABC):
    def __init__(self):
        self.n = None
        self.H = None
        self.H_batched = None
        self.psi0 = None
        self.psi0_batched = None
        self.exp_ops = None

    @abstractmethod
    def t_save(self, n: int) -> Tensor:
        pass

    def psi(self, t: float) -> Tensor:
        raise NotImplementedError

    def psis(self, t: Tensor) -> Tensor:
        return torch.stack([self.psi(t_.item()) for t_ in t])


class Cavity(ClosedSystem):
    # `H_batched: (3, n, n)
    # `psi0_batched`: (4, n, n)
    # `exp_ops`: (2, n, n)

    def __init__(
        self, *, n: int, delta: float, alpha0: complex, requires_grad: bool = False
    ):
        # store parameters
        self.n = n
        self.delta = torch.as_tensor(delta).requires_grad_(requires_grad)
        self.alpha0 = torch.as_tensor(alpha0).requires_grad_(requires_grad)

        # bosonic operators
        a = tq.destroy(self.n)
        adag = a.adjoint()

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.exp_ops = [(a + adag) / sqrt(2), 1j * (adag - a) / sqrt(2)]

        # prepare initial states
        self.psi0 = tq.coherent(self.n, self.alpha0)
        self.psi0_batched = [
            tq.coherent(self.n, self.alpha0),
            tq.coherent(self.n, 1j * self.alpha0),
            tq.coherent(self.n, -self.alpha0),
            tq.coherent(self.n, -1j * self.alpha0),
        ]

    def t_save(self, num_t_save: int) -> Tensor:
        """Compute t_save"""
        t_end = 2 * pi / self.delta  # a full rotation
        return torch.linspace(0.0, t_end, num_t_save)

    def psi(self, t: float) -> Tensor:
        """Compute the exact state vector at a given time."""
        alpha_t = self.alpha0 * torch.exp(-1j * self.delta * t)
        return tq.coherent(self.n, alpha_t)
