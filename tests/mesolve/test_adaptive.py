import dynamiqs as dq

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestAdaptive(SolverTester):
    def test_batching(self):
        options = dq.options.Dopri5()
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Dopri5()
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Dopri5(gradient_alg='autograd')
        self._test_gradient(options, grad_leaky_cavity_8, num_t_save=11)
