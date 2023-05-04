import functools

import torch
from torch import Tensor

from torchqdynamics.options import ODEFixedStep
from torchqdynamics.solvers.ode.forward_solver import ForwardSolver
from torchqdynamics.solvers.solver import depends_on_H


class FakeSolver(ForwardSolver):
    def __init__(self, H):
        y = torch.zeros((3, 3))
        t_save = torch.linspace(0, 10, 100)
        exp_ops = torch.zeros((3, 3))
        options = ODEFixedStep(dt=1e-3)

        super().__init__(
            H, y, t_save, exp_ops, options, gradient_alg=None, parameters=None
        )
        self.called_1 = 0
        self.called_2 = 0

    def forward(self, t: float, y: Tensor) -> Tensor:
        v1 = self.variable_1(t)
        v2 = self.variable_2(t)
        return torch.ones_like(y) * v1 * v2

    @depends_on_H
    def variable_1(self, t):
        """Function that only depends on H"""
        _ = self.H(t)
        self.called_1 += 1
        return self.called_1

    @depends_on_H
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
