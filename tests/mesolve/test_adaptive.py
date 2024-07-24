import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.solver import Tsit5

from ..integrator_tester import IntegratorTester
from .open_system import ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast


class TestMESolveAdaptive(IntegratorTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)
