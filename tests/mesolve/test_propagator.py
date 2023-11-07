from dynamiqs.gradient import Autograd
from dynamiqs.solver import Propagator

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(leaky_cavity_8, Propagator())

    def test_correctness(self):
        self._test_correctness(leaky_cavity_8, Propagator(), num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_leaky_cavity_8, Propagator(), Autograd(), num_tsave=11)
