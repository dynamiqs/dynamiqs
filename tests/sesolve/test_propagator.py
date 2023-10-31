from dynamiqs.solver import Propagator

from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestSEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(cavity_8, Propagator())

    def test_correctness(self):
        self._test_correctness(cavity_8, Propagator(), num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_cavity_8, Propagator().autograd(), num_tsave=11)
