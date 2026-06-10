import pytest

from dynamiqs.gradient import HigherOrder
from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, dia_ocavity

# we only test Tsit5 to keep the unit test suite fast


@pytest.mark.run(order=TEST_LONG)
class TestMESolveHessian(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity])
    def test_hessian(self, system):
        self._test_hessian(
            system, Tsit5(rtol=1e-8, atol=1e-8), HigherOrder(), rtol=1e-3, atol=1e-4
        )
