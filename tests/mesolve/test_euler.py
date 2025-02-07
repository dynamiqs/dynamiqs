import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.method import Euler

from ..order import TEST_LONG
from ..solver_tester import SolverTester
from .open_system import dense_ocavity, dia_ocavity, otdqubit


@pytest.mark.run(order=TEST_LONG)
class TestMESolveEuler(SolverTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    def test_correctness(self, system):
        method = Euler(dt=1e-4)
        self._test_correctness(system, method, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        method = Euler(dt=1e-4)
        self._test_gradient(system, method, gradient, rtol=1e-2, atol=1e-2)
