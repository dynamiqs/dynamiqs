from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestPropagator(SolverTester):
    def test_batching(self):
        options = dict()
        self._test_batching('propagator', options, cavity_8)

    def test_correctness(self):
        options = dict()
        self._test_correctness('propagator', options, cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dict(gradient_alg='autograd')
        self._test_gradient('propagator', options, grad_cavity_8, num_t_save=11)
