import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.method import Expm

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(dense_ocavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_ocavity, Expm(), Autograd())
