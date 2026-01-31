import pytest

from dynamiqs import Options
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, dia_ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptive(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize(
        ('vectorized', 'assume_hermitian'),
        [(True, True), (False, True), (False, False)],
    )
    def test_correctness(self, system, vectorized, assume_hermitian):
        options = Options(vectorized=vectorized, assume_hermitian=assume_hermitian)
        self._test_correctness(system, Tsit5(), options=options)

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)
