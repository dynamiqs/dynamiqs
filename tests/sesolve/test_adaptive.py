from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8, grad_tdqubit, tdqubit


class TestSEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(cavity_8, Dopri5())

    def test_correctness(self):
        self._test_correctness(cavity_8, Dopri5(), num_tsave=11)

    def test_td_correctness(self):
        self._test_correctness(tdqubit, Dopri5(), num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_cavity_8, Dopri5(), Autograd(), num_tsave=11)

    def test_td_autograd(self):
        self._test_gradient(grad_tdqubit, Dopri5(), Autograd(), num_tsave=11)
