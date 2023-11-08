import pytest

from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Rouchon1, Rouchon2

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMERouchon1(SolverTester):
    def test_batching(self):
        solver = Rouchon1(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_correctness(self, normalize):
        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_correctness(
            leaky_cavity_8,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-4,
            exp_save_atol=1e-2,
        )

    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_autograd(self, normalize):
        solver = Rouchon1(dt=1e-3, normalize=normalize)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            Autograd(),
            num_tsave=11,
            rtol=1e-4,
            atol=1e-2,
        )

    @pytest.mark.parametrize('normalize', [None, 'sqrt', 'cholesky'])
    def test_adjoint(self, normalize):
        solver = Rouchon1(dt=1e-3, normalize=normalize)
        gradient = Adjoint(params=grad_leaky_cavity_8.parameters)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            gradient,
            num_tsave=11,
            rtol=1e-4,
            atol=1e-2,
        )


class TestMERouchon2(SolverTester):
    def test_batching(self):
        solver = Rouchon2(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    def test_correctness(self):
        solver = Rouchon2(dt=1e-3)
        self._test_correctness(
            leaky_cavity_8,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-2,
        )

    def test_autograd(self):
        solver = Rouchon2(dt=1e-3)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            Autograd(),
            num_tsave=11,
        )

    def test_adjoint(self):
        solver = Rouchon2(dt=1e-3)
        gradient = Adjoint(params=grad_leaky_cavity_8.parameters)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            gradient,
            num_tsave=11,
            atol=1e-4,
        )
