import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd, ForwardAutograd
from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .closed_system import dense_cavity, dia_cavity, tdqubit

# we only test Tsit5 to keep the unit test suite fast


@pytest.mark.run(order=TEST_LONG)
class TestSESolveAdaptive(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    @pytest.mark.parametrize(
        'gradient', [Autograd(), CheckpointAutograd(), ForwardAutograd()]
    )
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)
