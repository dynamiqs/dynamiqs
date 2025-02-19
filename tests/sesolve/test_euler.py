import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.method import Euler

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .closed_system import dense_cavity, dia_cavity, tdqubit


@pytest.mark.run(order=TEST_LONG)
class TestSESolveEuler(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    def test_correctness(self, system):
        method = Euler(dt=1e-4)
        self._test_correctness(system, method, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        method = Euler(dt=1e-4)
        self._test_gradient(system, method, gradient, rtol=1e-2, atol=1e-2)
