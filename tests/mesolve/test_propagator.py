from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..solver_tester import OpenSolverTester
from .open_system import gocavity, ocavity


class TestMEPropagator(OpenSolverTester):
    def test_batching(self):
        self._test_batching(ocavity, Propagator())
        self._test_flat_batching(ocavity, Propagator())

    def test_correctness(self):
        self._test_correctness(ocavity, Propagator())

    def test_autograd(self):
        self._test_gradient(gocavity, Propagator(), Autograd())
