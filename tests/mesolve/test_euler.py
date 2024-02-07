import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .open_system import ocavity


class TestMEEuler(SolverTester):
    # @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('system', [ocavity])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    # @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('system', [ocavity])
    def test_autograd(self, system):
        solver = Euler(dt=1e-4)
        self._test_gradient(system, solver, Autograd(), rtol=1e-2, atol=1e-2)
