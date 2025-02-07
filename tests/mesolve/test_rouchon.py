import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.method import Rouchon1

from ..order import TEST_LONG
from ..solver_tester import SolverTester
from .open_system import dense_ocavity, dia_ocavity, otdqubit


@pytest.mark.run(order=TEST_LONG)
class TestMESolveRouchon1(SolverTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('normalize', [True, False])
    def test_correctness(self, system, normalize):
        method = Rouchon1(dt=1e-4, normalize=normalize)
        self._test_correctness(system, method, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('normalize', [True, False])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, normalize, gradient):
        method = Rouchon1(dt=1e-4, normalize=normalize)
        self._test_gradient(system, method, gradient, rtol=1e-3, atol=1e-3)
