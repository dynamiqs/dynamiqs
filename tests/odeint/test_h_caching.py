import functools

import torch
from torch import Tensor

from torchqdynamics.odeint import ForwardQSolver, H_dependent


class FakeSolver(ForwardQSolver):
    def __init__(self, H):
        super().__init__(H)
        self.called_1 = 0
        self.called_2 = 0

    def forward(self, t: float, y: Tensor) -> Tensor:
        v1 = self.variable_1(t)
        v2 = self.variable_2(t)
        return torch.ones_like(y) * v1 * v2

    @H_dependent
    def variable_1(self, t):
        """Function that only depends on H"""
        _ = self.H(t)
        self.called_1 += 1
        return self.called_1

    @H_dependent
    def variable_2(self, t):
        """Function that depends on H and `variable_1`"""
        _ = self.H(t)
        self.called_2 += 1
        return self.called_1 * self.called_2


def test_H_caching():
    H = torch.eye(3)
    y = torch.eye(3)

    H_called = [0]

    def compute_H(H_called, t):
        H_called[0] += 1
        return t * H

    compute_H = functools.partial(compute_H, H_called)

    solver = FakeSolver(compute_H)
    solver.forward(1.0, y)
    assert solver.called_1 == 1
    assert solver.called_2 == 1
    assert H_called[0] == 1

    solver.forward(2.0, y)
    assert solver.called_1 == 2
    assert solver.called_2 == 2
    assert H_called[0] == 2
