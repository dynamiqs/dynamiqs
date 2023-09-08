from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestAdaptive(SolverTester):
    def test_batching(self):
        options = dict()
        self._test_batching('dopri5', options, cavity_8)

    def test_correctness(self):
        options = dict()
        self._test_correctness('dopri5', options, cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dict(gradient_alg='autograd')
        self._test_gradient('dopri5', options, grad_cavity_8, num_t_save=11)
