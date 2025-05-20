import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd, ForwardAutograd
from dynamiqs.method import Rouchon1, Rouchon2, Rouchon3

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - exact_expm: set to False
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveRouchon(IntegratorTester):
    @pytest.mark.parametrize('method_class', [Rouchon1, Rouchon2, Rouchon3])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, method_class, system):
        method = method_class(dt=1e-4)
        self._test_correctness(system, method, esave_atol=1e-3)

    @pytest.mark.parametrize('method_class', [Rouchon1, Rouchon2, Rouchon3])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    @pytest.mark.parametrize(
        'gradient', [Autograd(), CheckpointAutograd(), ForwardAutograd()]
    )
    def test_gradient(self, method_class, system, gradient):
        method = method_class(dt=1e-4)
        self._test_gradient(system, method, gradient, rtol=1e-3, atol=1e-3)
