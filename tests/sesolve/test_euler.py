import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity, tdqubit


class TestSEEuler(SolverTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Euler(dt=1e-4), esave_atol=1e-3)

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Euler(dt=1e-4), Autograd(), rtol=1e-2, atol=1e-2)
