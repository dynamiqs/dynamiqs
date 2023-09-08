from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(cavity_8, 'propagator')

    def test_correctness(self):
        self._test_correctness(cavity_8, 'propagator', num_t_save=11)

    def test_autograd(self):
        self._test_gradient(grad_cavity_8, 'propagator', 'autograd', num_t_save=11)
