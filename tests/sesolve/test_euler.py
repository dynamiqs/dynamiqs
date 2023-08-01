import dynamiqs as dq

from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestSEEuler(SolverTester):
    def test_batching(self):
        options = dq.options.Euler(dt=1e-2)
        self._test_batching(options, cavity_8)

    def test_correctness(self):
        options = dq.options.Euler(dt=1e-4)
        self._test_correctness(options, cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Euler(dt=1e-4, gradient_alg='autograd')
        self._test_gradient(options, grad_cavity_8, num_t_save=11, rtol=1e-1, atol=1e-2)
