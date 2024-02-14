import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Tsit5

from ..solver_tester import SolverTester
from .closed_system import cavity, tdqubit

# we only test Tsit5 to keep the unit test suite fast


class TestSEAdaptive(SolverTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Tsit5(), Autograd())
