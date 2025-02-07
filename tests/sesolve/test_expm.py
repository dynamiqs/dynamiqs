import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.method import Expm

from ..order import TEST_LONG
from ..solver_tester import SolverTester
from .closed_system import dense_cavity


@pytest.mark.run(order=TEST_LONG)
class TestSESolveExpm(SolverTester):
    def test_correctness(self):
        self._test_correctness(dense_cavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_cavity, Expm(), Autograd())
