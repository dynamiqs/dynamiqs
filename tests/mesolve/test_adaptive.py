import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .open_system import gocavity, gotdqubit, ocavity, otdqubit


class TestMEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(ocavity, Dopri5())

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())
