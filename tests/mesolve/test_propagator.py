from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..integrator_tester import IntegratorTester
from .open_system import ocavity


class TestMESolvePropagator(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(ocavity, Propagator())

    def test_gradient(self):
        self._test_gradient(ocavity, Propagator(), Autograd())
