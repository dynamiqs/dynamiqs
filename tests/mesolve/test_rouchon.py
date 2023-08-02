import pytest

import dynamiqs as dq

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMERouchon1(SolverTester):
    def test_batching(self):
        options = dq.options.Rouchon1(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    @pytest.mark.parametrize('sqrt_normalization', [False, True])
    def test_correctness(self, sqrt_normalization):
        options = dq.options.Rouchon1(dt=1e-3, sqrt_normalization=sqrt_normalization)
        self._test_correctness(
            options,
            leaky_cavity_8,
            num_t_save=11,
            y_save_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-2,
        )

    @pytest.mark.parametrize('sqrt_normalization', [False, True])
    def test_autograd(self, sqrt_normalization):
        options = dq.options.Rouchon1(
            dt=1e-3, sqrt_normalization=sqrt_normalization, gradient_alg='autograd'
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize('sqrt_normalization', [False, True])
    def test_adjoint(self, sqrt_normalization):
        options = dq.options.Rouchon1(
            dt=1e-3,
            sqrt_normalization=sqrt_normalization,
            gradient_alg='adjoint',
            parameters=grad_leaky_cavity_8.parameters,
        )
        self._test_gradient(
            options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )


class TestMERouchon2(SolverTester):
    def test_batching(self):
        options = dq.options.Rouchon2(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Rouchon2(dt=1e-3)
        self._test_correctness(
            options,
            leaky_cavity_8,
            num_t_save=11,
            y_save_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-4,
        )

    def test_autograd(self):
        options = dq.options.Rouchon2(dt=1e-3, gradient_alg='autograd')
        self._test_gradient(options, grad_leaky_cavity_8, num_t_save=11)

    def test_adjoint(self):
        options = dq.options.Rouchon2(
            dt=1e-3, gradient_alg='adjoint', parameters=grad_leaky_cavity_8.parameters
        )
        self._test_gradient(options, grad_leaky_cavity_8, num_t_save=11, atol=1e-4)
