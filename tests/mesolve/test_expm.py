from dynamiqs.gradient import Autograd
from dynamiqs.solver import Expm

from ..integrator_tester import IntegratorTester
from .open_system import ocavity


class TestMESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(ocavity, Expm())

    def test_gradient(self):
        self._test_gradient(ocavity, Expm(), Autograd())
