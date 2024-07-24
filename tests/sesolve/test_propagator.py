from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..integrator_tester import IntegratorTester
from .closed_system import cavity


class TestSESolvePropagator(IntegratorTester):
    def test_correctness(self):
        self._test_correctness(cavity, Propagator())

    def test_gradient(self):
        self._test_gradient(cavity, Propagator(), Autograd())
