import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Rouchon1, Rouchon2

from ..solver_tester import SolverTester
from .open_system import (
    damped_tdqubit,
    grad_damped_tdqubit,
    grad_leaky_cavity_8,
    leaky_cavity_8,
)


class TestMERouchon1(SolverTester):
    def test_batching(self):
        solver = Rouchon1(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    @pytest.mark.parametrize(
        'system, exp_save_rtol', [(leaky_cavity_8, 1e-4), (damped_tdqubit, 1e-2)]
    )
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_correctness(self, system, exp_save_rtol, normalize):
        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_correctness(
            system,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=exp_save_rtol,
            exp_save_atol=1e-2,
        )

    @pytest.mark.parametrize('system', [grad_leaky_cavity_8, grad_damped_tdqubit])
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_autograd(self, system, normalize):
        if system is grad_damped_tdqubit and normalize == 'sqrt':
            pytest.skip('sqrt normalization broken for TD system gradient computation')

        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_gradient(
            system, solver, Autograd(), num_tsave=11, rtol=1e-4, atol=1e-2
        )

    @pytest.mark.parametrize('system', [grad_leaky_cavity_8, grad_damped_tdqubit])
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_adjoint(self, system, normalize):
        if system is grad_damped_tdqubit and normalize == 'sqrt':
            pytest.skip('sqrt normalization broken for TD system gradient computation')

        solver = Rouchon1(dt=1e-3, normalize=normalize)
        gradient = Adjoint(params=system.params)
        self._test_gradient(
            system, solver, gradient, num_tsave=11, rtol=1e-4, atol=1e-2
        )


class TestMERouchon2(SolverTester):
    def test_batching(self):
        solver = Rouchon2(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    @pytest.mark.parametrize('system', [leaky_cavity_8, damped_tdqubit])
    def test_correctness(self, system):
        solver = Rouchon2(dt=1e-3)
        self._test_correctness(
            system,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-2,
        )

    @pytest.mark.parametrize(
        'system,rtol,atol',
        [(grad_leaky_cavity_8, 1e-3, 1e-5), (grad_damped_tdqubit, 1e-2, 1e-3)],
    )
    def test_autograd(self, system, rtol, atol):
        solver = Rouchon2(dt=1e-3)
        self._test_gradient(
            system, solver, Autograd(), num_tsave=11, rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize(
        'system,rtol,atol',
        [(grad_leaky_cavity_8, 1e-2, 1e-5), (grad_damped_tdqubit, 1e-2, 1e-3)],
    )
    def test_adjoint(self, system, rtol, atol):
        solver = Rouchon2(dt=1e-3)
        gradient = Adjoint(params=system.params)
        self._test_gradient(
            system, solver, gradient, num_tsave=11, rtol=rtol, atol=atol
        )
