from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..solver_tester import SolverTester
from .open_system import ocavity


class TestMEPropagator(SolverTester):
    def test_correctness(self):
        self._test_correctness(ocavity, Propagator())

    def test_gradient(self):
        self._test_gradient(ocavity, Propagator(), Autograd())
