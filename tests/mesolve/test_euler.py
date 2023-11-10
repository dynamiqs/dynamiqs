import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .open_system import (
    damped_tdqubit,
    grad_damped_tdqubit,
    grad_leaky_cavity_8,
    leaky_cavity_8,
)


class TestMEEuler(SolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    @pytest.mark.parametrize('system', [leaky_cavity_8, damped_tdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            system,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-3,
        )

    @pytest.mark.parametrize('system', [grad_leaky_cavity_8, grad_damped_tdqubit])
    def test_autograd(self, system):
        solver = Euler(dt=1e-3)
        self._test_gradient(
            system, solver, Autograd(), num_tsave=11, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize('system', [grad_leaky_cavity_8, grad_damped_tdqubit])
    def test_adjoint(self, system):
        solver = Euler(dt=1e-3)
        gradient = Adjoint(params=system.params)
        self._test_gradient(
            system, solver, gradient, num_tsave=11, rtol=1e-3, atol=1e-2
        )
