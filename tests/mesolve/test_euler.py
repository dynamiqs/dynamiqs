import pytest

from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .open_system import ocavity, otdqubit


class TestMEEuler(SolverTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(system, solver, esave_atol=1e-3)

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('autograd', [True, False])
    def test_gradient(self, system, autograd):
        solver = Euler(dt=1e-4, autograd=autograd)
        self._test_gradient(system, solver, rtol=1e-2, atol=1e-2)
