from dynamiqs.gradient import Autograd
from dynamiqs.solver import Expm

from ..integrator_tester import IntegratorTester
from .closed_system import cavity


class TestSESolveExpm(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(cavity, Expm())

    def test_gradient(self):
        self._test_gradient(cavity, Expm(), Autograd())
