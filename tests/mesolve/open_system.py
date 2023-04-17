from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import torch
from torch import Tensor

import torchqdynamics as tq


class OpenSystem(ABC):
    def __init__(self):
        self.n = None
        self.H = None
        self.H_batched = None
        self.jump_ops = None
        self.rho0 = None
        self.rho0_batched = None
        self.exp_ops = None

    @abstractmethod
    def t_save(self, n: int) -> Tensor:
        pass

    def rho(self, t: float) -> Tensor:
        raise NotImplementedError

    def rhos(self, t: Tensor) -> Tensor:
        return torch.stack([self.rho(t_.item()) for t_ in t])


class LeakyCavity(OpenSystem):
    # `H_batched: (3, n, n)
    # `jump_ops`: (2, n, n)
    # `rho0_batched`: (4, n, n)
    # `exp_ops`: (2, n, n)

    def __init__(self, *, n: int, kappa: float, delta: float, alpha0: complex):
        self.n, self.kappa, self.delta, self.alpha0 = n, kappa, delta, alpha0

        a = qt.destroy(n)
        adag = a.dag()

        self.H = delta * adag * a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.jump_ops = [np.sqrt(kappa) * a, np.eye(n)]
        self.exp_ops = [(a + adag) / np.sqrt(2), (a - adag) / (np.sqrt(2) * 1j)]

        self.rho0 = qt.coherent(n, alpha0)
        self.rho0_batched = [
            qt.coherent(n, alpha0),
            qt.coherent(n, 1j * alpha0),
            qt.coherent(n, -alpha0),
            qt.coherent(n, -1j * alpha0),
        ]

    def t_save(self, n: int) -> Tensor:
        t_end = 1 / (self.delta / (2 * np.pi))  # a full rotation
        return torch.linspace(0.0, t_end, n)

    def rho(self, t: float) -> Tensor:
        alpha_t = (
            self.alpha0 * np.exp(-1j * self.delta * t) * np.exp(-0.5 * self.kappa * t)
        )
        rho_t = qt.coherent(self.n, alpha_t)
        return tq.ket_to_dm(tq.from_qutip(rho_t))
