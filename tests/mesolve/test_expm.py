from dynamiqs.gradient import Autograd
from dynamiqs.solver import Expm

from ..integrator_tester import IntegratorTester
from .open_system import dense_ocavity


class TestMESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(dense_ocavity, Expm())

    def test_gradient(self):
        self._test_gradient(dense_ocavity, Expm(), Autograd())
