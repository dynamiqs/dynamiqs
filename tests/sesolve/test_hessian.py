import pytest

from dynamiqs.gradient import HigherOrder
from dynamiqs.method import Dopri5, Dopri8, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_cavity, dia_cavity, tdqubit

# the explicit adaptive ODE methods; `Euler`'s HigherOrder support is checked at the
# gate level in tests/core/test_gradient_support.py (its fixed-step Hessian is too
# inaccurate for a meaningful analytical comparison here)


@pytest.mark.run(order=TEST_LONG)
class TestSESolveHessian(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    @pytest.mark.parametrize('method', [Dopri5(), Dopri8(), Tsit5()])
    def test_hessian(self, system, method):
        self._test_hessian(system, method, HigherOrder(), rtol=1e-3, atol=2e-4)
