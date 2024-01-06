import pytest
from dynamiqs.gradient import Autograd

from ..solver_tester import SolverTester
from .open_system import gocavity, ocavity


Propagator = None


@pytest.mark.skip(reason="Propagator not implemented yet")
class TestMEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(ocavity, Propagator())

    def test_correctness(self):
        self._test_correctness(ocavity, Propagator())

    def test_autograd(self):
        self._test_gradient(gocavity, Propagator(), Autograd())
