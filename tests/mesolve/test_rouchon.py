from math import pi

import dynamiqs as dq

from .me_solver_tester import MEGradientSolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0)
grad_leaky_cavity_8 = LeakyCavity(
    n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0, requires_grad=True
)


class TestMERouchon1(MEGradientSolverTester):
    def test_batching(self):
        options = dq.options.Rouchon1(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Rouchon1(dt=1e-3)
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)

        options = dq.options.Rouchon1(dt=1e-3, sqrt_normalization=True)
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Rouchon1(dt=1e-3, gradient_alg='autograd')
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )

        options = dq.options.Rouchon1(
            dt=1e-3, sqrt_normalization=True, gradient_alg='autograd'
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )

    def test_adjoint(self):
        options = dq.options.Rouchon1(
            dt=1e-3, gradient_alg='adjoint', parameters=grad_leaky_cavity_8.parameters
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=5e-2, atol=3e-3
        )

        options = dq.options.Rouchon1(
            dt=1e-3,
            sqrt_normalization=True,
            gradient_alg='adjoint',
            parameters=grad_leaky_cavity_8.parameters,
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=5e-2, atol=3e-3
        )


class TestMERouchon2(MEGradientSolverTester):
    def test_batching(self):
        options = dq.options.Rouchon2(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Rouchon2(dt=1e-3)
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)

    def test_autograd(self):
        options = dq.options.Rouchon2(dt=1e-3, gradient_alg='autograd')
        self._test_gradient(options, grad_leaky_cavity_8, num_t_save=11)

    def test_adjoint(self):
        options = dq.options.Rouchon2(
            dt=1e-3, gradient_alg='adjoint', parameters=grad_leaky_cavity_8.parameters
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=5e-2, atol=3e-3
        )
