from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
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

    def __init__(self, *, n: int, delta: float, alpha0: complex):
        self.n = n
        self.delta = delta
        self.alpha0 = alpha0

        a = qt.destroy(n)
        adag = a.dag()

        self.H = delta * adag * a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.exp_ops = [(a + adag) / np.sqrt(2), (a - adag) / (np.sqrt(2) * 1j)]

        self.psi0 = qt.coherent(n, alpha0)
        self.psi0_batched = [
            qt.coherent(n, alpha0),
            qt.coherent(n, 1j * alpha0),
            qt.coherent(n, -alpha0),
            qt.coherent(n, -1j * alpha0),
        ]

    def t_save(self, n: int) -> Tensor:
        t_end = self.delta / (2 * np.pi)  # a full rotation
        return torch.linspace(0.0, t_end, n)

    def psi(self, t: float) -> Tensor:
        alpha_t = self.alpha0 * np.exp(-1j * self.delta * t)
        psi_t = qt.coherent(self.n, alpha_t)
        return tq.from_qutip(psi_t)


cavity_8 = Cavity(n=8, delta=2 * np.pi, alpha0=1.0)
