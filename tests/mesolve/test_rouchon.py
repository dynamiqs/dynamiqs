import pytest

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMERouchon1(SolverTester):
    def test_batching(self):
        options = dict(dt=1e-2)
        self._test_batching(leaky_cavity_8, 'rouchon1', options=options)

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_correctness(self, cholesky_normalization):
        options = dict(dt=1e-3, cholesky_normalization=cholesky_normalization)
        self._test_correctness(
            leaky_cavity_8,
            'rouchon1',
            options=options,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-4,
            exp_save_atol=1e-2,
        )

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_autograd(self, cholesky_normalization):
        options = dict(dt=1e-3, cholesky_normalization=cholesky_normalization)
        self._test_gradient(
            grad_leaky_cavity_8,
            'rouchon1',
            'autograd',
            options=options,
            num_tsave=11,
            rtol=1e-4,
            atol=1e-2,
        )

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_adjoint(self, cholesky_normalization):
        options = dict(
            dt=1e-3,
            cholesky_normalization=cholesky_normalization,
            parameters=grad_leaky_cavity_8.parameters,
        )
        self._test_gradient(
            grad_leaky_cavity_8,
            'rouchon1',
            'adjoint',
            options=options,
            num_tsave=11,
            rtol=1e-4,
            atol=1e-2,
        )


class TestMERouchon2(SolverTester):
    def test_batching(self):
        options = dict(dt=1e-2)
        self._test_batching(leaky_cavity_8, 'rouchon2', options=options)

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_correctness(self, cholesky_normalization):
        options = dict(dt=1e-3, cholesky_normalization=cholesky_normalization)
        self._test_correctness(
            leaky_cavity_8,
            'rouchon2',
            options=options,
            num_tsave=11,
            ysave_norm_atol=1e-4,
            exp_save_rtol=1e-4,
            exp_save_atol=1e-4,
        )

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_autograd(self, cholesky_normalization):
        options = dict(dt=1e-3, cholesky_normalization=cholesky_normalization)
        self._test_gradient(
            grad_leaky_cavity_8,
            'rouchon2',
            gradient='autograd',
            options=options,
            num_tsave=11,
        )

    @pytest.mark.parametrize('cholesky_normalization', [False, True])
    def test_adjoint(self, cholesky_normalization):
        options = dict(
            dt=1e-3,
            cholesky_normalization=cholesky_normalization,
            parameters=grad_leaky_cavity_8.parameters,
        )
        self._test_gradient(
            grad_leaky_cavity_8,
            'rouchon2',
            'adjoint',
            options=options,
            num_tsave=11,
            atol=1e-4,
        )
