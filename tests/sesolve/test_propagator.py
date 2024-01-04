from dynamiqs.gradient import Autograd
from dynamiqs.solvers import Propagator

from ..solver_tester import SolverTester
from .closed_system import cavity, gcavity


class TestSEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(cavity, Propagator())

    def test_correctness(self):
        self._test_correctness(cavity, Propagator())

    def test_autograd(self):
        self._test_gradient(gcavity, Propagator(), Autograd())
