import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8, grad_tdqubit, tdqubit


class TestSEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(cavity_8, Dopri5())

    @pytest.mark.parametrize('system', [cavity_8, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5(), num_tsave=11)

    @pytest.mark.parametrize('system', [grad_cavity_8, grad_tdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd(), num_tsave=11)
