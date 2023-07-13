from math import pi

import dynamiqs as dq

from .closed_system import Cavity
from .se_solver_tester import SEAutogradSolverTester

cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0)
grad_cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0, requires_grad=True)


class TestSEEuler(SEAutogradSolverTester):
    def test_batching(self):
        options = dq.options.Euler(dt=1e-2)
        self._test_batching(options, cavity_8)

    def test_correctness(self):
        options = dq.options.Euler(dt=1e-4)
        self._test_correctness(options, cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Euler(dt=1e-4, gradient_alg='autograd')
        self._test_autograd(options, grad_cavity_8, num_t_save=11, rtol=1e-1, atol=1e-2)
