from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..solver_tester import SolverTester
from .closed_system import cavity


class TestSEPropagator(SolverTester):
    def test_correctness(self):
        self._test_correctness(cavity, Propagator())

    def test_gradient(self):
        self._test_gradient(cavity, Propagator(), Autograd())
