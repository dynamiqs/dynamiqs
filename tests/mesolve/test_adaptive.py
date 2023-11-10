import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .open_system import (
    damped_tdqubit,
    grad_damped_tdqubit,
    grad_leaky_cavity_8,
    leaky_cavity_8,
)


class TestMEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(leaky_cavity_8, Dopri5())

    @pytest.mark.parametrize('system', [leaky_cavity_8, damped_tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5(), num_tsave=11)

    @pytest.mark.parametrize('system', [grad_leaky_cavity_8, grad_damped_tdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd(), num_tsave=11)
