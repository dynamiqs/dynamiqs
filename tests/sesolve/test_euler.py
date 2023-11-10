import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity, gcavity, gtdqubit, tdqubit


class TestSEEuler(SolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(cavity, solver)

    @pytest.mark.parametrize('system,tol', [(cavity, 1e-2), (tdqubit, 1e-3)])
    def test_correctness(self, system, tol):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            system, solver, ysave_atol=tol, esave_rtol=tol, esave_atol=tol
        )

    @pytest.mark.parametrize('system,rtol', [(gcavity, 5e-2), (gtdqubit, 1e-2)])
    def test_autograd(self, system, rtol):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, Autograd(), rtol=rtol, atol=1e-2)
