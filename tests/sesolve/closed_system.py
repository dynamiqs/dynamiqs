from abc import ABC, abstractmethod
from math import pi, sqrt

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.options import Options
from dynamiqs.solvers.result import Result
from dynamiqs.utils.tensor_types import ArrayLike


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
        """Compute the save time tensor."""
        pass

    def psi(self, t: float) -> Tensor:
        """Compute the exact state vector at a given time."""
        raise NotImplementedError

    def psis(self, t: Tensor) -> Tensor:
        return torch.stack([self.psi(t_.item()) for t_ in t])

    def expect(self, t: float) -> Tensor:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Tensor) -> Tensor:
        return torch.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def sesolve(self, t_save: ArrayLike, options: Options) -> Result:
        return dq.sesolve(
            self.H,
            self.psi0,
            t_save,
            exp_ops=self.exp_ops,
            options=options,
        )


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
        a = dq.destroy(self.n)
        adag = a.mH

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.exp_ops = [(a + adag) / sqrt(2), 1j * (adag - a) / sqrt(2)]

        # prepare initial states
        self.psi0 = dq.coherent(self.n, self.alpha0)
        self.psi0_batched = [
            dq.coherent(self.n, self.alpha0),
            dq.coherent(self.n, 1j * self.alpha0),
            dq.coherent(self.n, -self.alpha0),
            dq.coherent(self.n, -1j * self.alpha0),
        ]

    def t_save(self, num_t_save: int) -> Tensor:
        t_end = 2 * pi / self.delta  # a full rotation
        return torch.linspace(0.0, t_end, num_t_save)

    def _alpha(self, t: float) -> Tensor:
        return self.alpha0 * torch.exp(-1j * self.delta * t)

    def psi(self, t: float) -> Tensor:
        return dq.coherent(self.n, self._alpha(t))

    def expect(self, t: float) -> Tensor:
        alpha_t = self._alpha(t)
        exp_x = sqrt(2) * alpha_t.real
        exp_p = sqrt(2) * alpha_t.imag
        return torch.tensor([exp_x, exp_p], dtype=alpha_t.dtype)
