import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.solver import Rouchon1

from ..integrator_tester import IntegratorTester
from .open_system import ocavity, otdqubit


class TestMESolveRouchon1(IntegratorTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        solver = Rouchon1(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        solver = Rouchon1(dt=1e-4)
        self._test_gradient(system, solver, gradient, rtol=1e-3, atol=1e-3)
