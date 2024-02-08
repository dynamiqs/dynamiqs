import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .closed_system import cavity


class TestSEEuler(SolverTester):
    @pytest.mark.skip(reason='broken test')
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(cavity, solver)

    # @pytest.mark.parametrize('system', [cavity, tdqubit])
    @pytest.mark.parametrize('system', [cavity])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    # @pytest.mark.parametrize('system', [cavity, tdqubit])
    @pytest.mark.parametrize('system', [cavity])
    def test_autograd(self, system):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, Autograd(), rtol=1e-2, atol=1e-2)
