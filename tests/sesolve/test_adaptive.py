import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solvers import Dopri5

from ..solver_tester import SolverTester
from .closed_system import cavity, gcavity, gtdqubit, tdqubit


class TestSEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(cavity, Dopri5())

    # @pytest.mark.parametrize('system', [cavity, tdqubit])
    @pytest.mark.parametrize('system', [cavity])  # TODO: restore tdqubit
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    @pytest.mark.parametrize('system', [gcavity, gtdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())
