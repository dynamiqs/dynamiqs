import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .open_system import ocavity, otdqubit


class TestMEDopri5(SolverTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), Adjoint()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Dopri5(), gradient)
