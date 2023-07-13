from math import pi

import dynamiqs as dq

from .closed_system import Cavity
from .se_solver_tester import SEGradientSolverTester

cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0)
grad_cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0, requires_grad=True)


class TestPropagator(SEGradientSolverTester):
    def test_batching(self):
        options = dq.options.Propagator()
        self._test_batching(options, cavity_8)

    def test_correctness(self):
        options = dq.options.Propagator()
        self._test_correctness(options, cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Propagator(gradient_alg='autograd')
        self._test_gradient(options, grad_cavity_8, num_t_save=11)
