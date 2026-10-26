import pytest

from dynamiqs.gradient import HigherOrder
from dynamiqs.method import Dopri5, Dopri8, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, dia_ocavity, otdqubit

# the explicit adaptive ODE methods, matching tests/sesolve/test_hessian.py; `Euler`'s
# HigherOrder support is checked at the gate level in tests/core/test_gradient_support
# (its fixed-step Hessian is too inaccurate for a meaningful analytical comparison)


@pytest.mark.run(order=TEST_LONG)
class TestMESolveHessian(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('method', [Dopri5(), Dopri8(), Tsit5()])
    def test_hessian(self, system, method):
        self._test_hessian(system, method, HigherOrder(), rtol=1e-3, atol=2e-4)
