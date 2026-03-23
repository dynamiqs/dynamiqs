import pytest

from dynamiqs.gradient import Direct
from dynamiqs.method import Rouchon2, Rouchon3

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveRouchon(IntegratorTester):
    @pytest.mark.parametrize('method_class', [Rouchon2, Rouchon3])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, method_class, system):
        self._test_correctness(system, method_class())

    @pytest.mark.parametrize('method_class', [Rouchon2])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct()])
    def test_gradient(self, method_class, system, gradient):
        self._test_gradient(system, method_class(), gradient)
