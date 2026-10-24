import pytest

from dynamiqs.gradient import BackwardCheckpointed, Forward
from dynamiqs.method import Expm, Rouchon2, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_composite_ocavity, dia_composite_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveComposite(IntegratorTester):
    # the vectorized and `Expm` paths build the full Liouvillian, the others apply the
    # composite operators to the state mode by mode
    @pytest.mark.parametrize('system', [dense_composite_ocavity, dia_composite_ocavity])
    @pytest.mark.parametrize('vectorized', [False, True])
    def test_correctness(self, system, vectorized):
        self._test_correctness(system, Tsit5(), vectorized=vectorized)

    @pytest.mark.parametrize('method', [Expm(), Rouchon2(dt=1e-3)])
    def test_correctness_methods(self, method):
        self._test_correctness(dense_composite_ocavity, method)

    @pytest.mark.parametrize('gradient', [BackwardCheckpointed(), Forward()])
    def test_gradient(self, gradient):
        self._test_gradient(dia_composite_ocavity, Tsit5(), gradient)
