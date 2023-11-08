from dynamiqs.gradient import Adjoint, Autograd
from dynamiqs.solver import Euler

from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMEEuler(SolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(leaky_cavity_8, solver)

    def test_correctness(self):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            leaky_cavity_8,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-3,
        )

    def test_autograd(self):
        solver = Euler(dt=1e-3)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            Autograd(),
            num_tsave=11,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_adjoint(self):
        solver = Euler(dt=1e-3)
        gradient = Adjoint(params=grad_leaky_cavity_8.parameters)
        self._test_gradient(
            grad_leaky_cavity_8,
            solver,
            gradient,
            num_tsave=11,
            rtol=1e-3,
            atol=1e-2,
        )
