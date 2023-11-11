import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Rouchon1, Rouchon2

from ..solver_tester import SolverTester
from .open_system import gocavity, gotdqubit, ocavity, otdqubit


class TestMERouchon1(SolverTester):
    def test_batching(self):
        solver = Rouchon1(dt=1e-2)
        self._test_batching(ocavity, solver)

    @pytest.mark.parametrize('system, esave_rtol', [(ocavity, 1e-4), (otdqubit, 1e-2)])
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_correctness(self, system, esave_rtol, normalize):
        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_correctness(
            system, solver, ysave_atol=1e-2, esave_rtol=esave_rtol, esave_atol=1e-2
        )

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_autograd(self, system, normalize):
        if system is gotdqubit and normalize == 'sqrt':
            pytest.skip('sqrt normalization broken for TD system gradient computation')

        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_gradient(system, solver, Autograd(), rtol=1e-4, atol=1e-2)

    @pytest.mark.parametrize('system', [gocavity, gotdqubit])
    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_adjoint(self, system, normalize):
        if system is gotdqubit and normalize == 'sqrt':
            pytest.skip('sqrt normalization broken for TD system gradient computation')

        solver = Rouchon1(dt=1e-3, normalize=normalize)
        gradient = Adjoint(params=system.params)
        self._test_gradient(system, solver, gradient, rtol=1e-4, atol=1e-2)


class TestMERouchon2(SolverTester):
    def test_batching(self):
        solver = Rouchon2(dt=1e-2)
        self._test_batching(ocavity, solver)

    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system):
        solver = Rouchon2(dt=1e-3)
        self._test_correctness(
            system, solver, ysave_atol=1e-2, esave_rtol=1e-2, esave_atol=1e-2
        )

    @pytest.mark.parametrize(
        'system,rtol,atol', [(gocavity, 1e-3, 1e-5), (gotdqubit, 1e-2, 1e-3)]
    )
    def test_autograd(self, system, rtol, atol):
        solver = Rouchon2(dt=1e-3)
        self._test_gradient(system, solver, Autograd(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        'system,rtol,atol', [(gocavity, 1e-2, 1e-4), (gotdqubit, 1e-2, 1e-3)]
    )
    def test_adjoint(self, system, rtol, atol):
        solver = Rouchon2(dt=1e-3)
        gradient = Adjoint(params=system.params)
        self._test_gradient(system, solver, gradient, rtol=rtol, atol=atol)
