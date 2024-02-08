import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .open_system import ocavity


class TestMEAdaptive(SolverTester):
    # @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('system', [ocavity])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    # @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('system', [ocavity])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())
