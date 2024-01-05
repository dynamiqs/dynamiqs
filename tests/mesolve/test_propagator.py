from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..solver_tester import SolverTester
from .open_system import gocavity, ocavity


class TestMEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(ocavity, Propagator())

    def test_correctness(self):
        self._test_correctness(ocavity, Propagator())

    def test_autograd(self):
        self._test_gradient(gocavity, Propagator(), Autograd())
