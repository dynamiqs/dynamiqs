import pytest

from dynamiqs.gradient import HigherOrder
from dynamiqs.method import Dopri5, Dopri8, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems.hessian import hqubit


@pytest.mark.run(order=TEST_LONG)
class TestSESolveHessian(IntegratorTester):
    @pytest.mark.parametrize('method', [Tsit5(), Dopri5(), Dopri8()])
    def test_hessian(self, method):
        self._test_hessian(hqubit, method, HigherOrder(), rtol=1e-2, atol=1e-2)
