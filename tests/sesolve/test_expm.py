import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.method import Expm

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .closed_system import dense_cavity


@pytest.mark.run(order=TEST_LONG)
class TestSESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(dense_cavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_cavity, Expm(), Autograd())
