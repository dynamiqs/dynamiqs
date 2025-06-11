import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd, ForwardAutograd
from dynamiqs.method import AdaptiveRouchon12, AdaptiveRouchon23

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - exact_expm: set to False
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveRouchon(IntegratorTester):
    @pytest.mark.parametrize('method_class', [AdaptiveRouchon12, AdaptiveRouchon23])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, method_class, system):
        # todo: changing ysave_atol and esave_rtol should not be necessary
        self._test_correctness(system, method_class(), ysave_atol=1e-2, esave_rtol=1e-2)

    @pytest.mark.parametrize('method_class', [AdaptiveRouchon12, AdaptiveRouchon23])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    @pytest.mark.parametrize(
        'gradient', [Autograd(), CheckpointAutograd(), ForwardAutograd()]
    )
    def test_gradient(self, method_class, system, gradient):
        # todo: changing rtol should not be necessary
        self._test_gradient(system, method_class(), gradient, rtol=1e-2)
