from math import pi

import dynamiqs as dq

from .me_solver_tester import MEGradientSolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0)
grad_leaky_cavity_8 = LeakyCavity(
    n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0, requires_grad=True
)


class TestMEEuler(MEGradientSolverTester):
    def test_batching(self):
        options = dq.options.Euler(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Euler(dt=1e-4)
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Euler(dt=1e-3, gradient_alg='autograd')
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )
