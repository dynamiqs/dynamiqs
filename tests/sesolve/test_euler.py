from dynamiqs.gradient import Autograd
from dynamiqs.solver import Euler

from ..solver_tester import ClosedSolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestSEEuler(ClosedSolverTester):
    def test_batching(self):
        solver = Euler(dt=1e-2)
        self._test_batching(cavity_8, solver)

    def test_correctness(self):
        solver = Euler(dt=1e-4)
        self._test_correctness(
            cavity_8,
            solver,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-1,
            exp_save_atol=1e-3,
        )

    def test_autograd(self):
        solver = Euler(dt=1e-4)
        self._test_gradient(
            grad_cavity_8,
            solver,
            Autograd(),
            num_tsave=11,
            rtol=1e-1,
            atol=1e-2,
        )
