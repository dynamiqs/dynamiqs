import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Euler

from ..solver_tester import OpenSolverTester
from .open_system import gocavity, gotdqubit, ocavity, otdqubit


class TestMEEuler(OpenSolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(ocavity, solver)

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            system, solver, ysave_atol=1e-2, esave_rtol=1e-2, esave_atol=1e-3
        )

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    def test_autograd(self, system):
        solver = Euler(dt=1e-3)
        self._test_gradient(system, solver, Autograd(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    def test_adjoint(self, system):
        solver = Euler(dt=1e-3)
        gradient = Adjoint(params=system.params)
        self._test_gradient(system, solver, gradient, rtol=1e-3, atol=1e-2)
