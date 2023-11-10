import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8, grad_tdqubit, tdqubit


class TestSEEuler(SolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(cavity_8, solver)

    @pytest.mark.parametrize('system,tol', [(cavity_8, 1e-2), (tdqubit, 1e-3)])
    def test_correctness(self, system, tol):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            system,
            solver,
            num_tsave=11,
            ysave_norm_atol=tol,
            exp_save_rtol=tol,
            exp_save_atol=tol,
        )

    @pytest.mark.parametrize(
        'system,rtol', [(grad_cavity_8, 5e-2), (grad_tdqubit, 1e-2)]
    )
    def test_autograd(self, system, rtol):
        solver = Euler(dt=1e-4)
        self._test_gradient(
            system, solver, Autograd(), num_tsave=11, rtol=rtol, atol=1e-2
        )
