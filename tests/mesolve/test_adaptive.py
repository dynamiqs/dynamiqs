import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Tsit5

from ..solver_tester import SolverTester
from .open_system import ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast


class TestMEAdaptive(SolverTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Tsit5(), Autograd())
