import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.solver import Euler

from ..integrator_tester import IntegratorTester
from .open_system import ocavity, otdqubit


class TestMESolveEuler(IntegratorTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, gradient, rtol=1e-2, atol=1e-2)
