import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .closed_system import cavity, gcavity, tdqubit


@pytest.mark.skip(reason='broken test')
class TestSEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(cavity, Dopri5())

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    def test_jit(self):
        self._test_jit(cavity, Dopri5())

    # @pytest.mark.parametrize('system', [gcavity, gtdqubit]) # TODO: restore gtdqubit
    @pytest.mark.parametrize('system', [gcavity])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())
