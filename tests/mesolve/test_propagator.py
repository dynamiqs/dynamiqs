from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(leaky_cavity_8, 'propagator')

    def test_correctness(self):
        self._test_correctness(leaky_cavity_8, 'propagator', num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_leaky_cavity_8, 'propagator', 'autograd', num_tsave=11)
