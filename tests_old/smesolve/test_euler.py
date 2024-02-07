import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from .monitored_system import gmcavity, mcavity
from .sme_solver_tester import SMESolverTester


@pytest.mark.skip(reason='broken test')
class TestSMEEuler(SMESolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(mcavity, solver)

    def test_correctness(self):
        solver = Euler(dt=1e-4)
        self._test_correctness(mcavity, solver, esave_atol=1e-3)

    def test_autograd(self):
        solver = Euler(dt=1e-4)
        self._test_gradient(gmcavity, solver, Autograd(), rtol=1e-2, atol=1e-2)
