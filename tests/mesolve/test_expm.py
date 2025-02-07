import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.method import Expm

from ..order import TEST_LONG
from ..solver_tester import SolverTester
from .open_system import dense_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveExpm(SolverTester):
    def test_correctness(self):
        self._test_correctness(dense_ocavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_ocavity, Expm(), Autograd())
