import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import OpenSolverTester
from .open_system import gocavity, gotdqubit, ocavity, otdqubit


class TestMEAdaptive(OpenSolverTester):
    def test_batching(self):
        self._test_batching(ocavity, Dopri5())

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    def test_adjoint(self, system):
        gradient = Adjoint(params=system.params)
        self._test_gradient(system, Dopri5(), gradient, atol=3e-3)
