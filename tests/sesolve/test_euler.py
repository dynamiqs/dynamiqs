import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity, gcavity, gtdqubit, tdqubit


class TestSEEuler(SolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(cavity, solver)

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    # @pytest.mark.parametrize('system', [gcavity, gtdqubit])
    @pytest.mark.parametrize('system', [gcavity])
    def test_autograd(self, system):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, Autograd(), rtol=1e-2, atol=1e-2)
