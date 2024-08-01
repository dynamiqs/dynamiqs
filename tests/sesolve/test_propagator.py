from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..integrator_tester import IntegratorTester
from .closed_system import dense_cavity


class TestSESolvePropagator(IntegratorTester):

    def test_correctness(self):
        self._test_correctness(dense_cavity, Propagator())

    def test_gradient(self):
        self._test_gradient(dense_cavity, Propagator(), Autograd())
