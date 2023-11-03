from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .open_system import (
    damped_tdqubit,
    grad_damped_tdqubit,
    grad_leaky_cavity_8,
    leaky_cavity_8,
)


class TestMEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(leaky_cavity_8, Dopri5())

    def test_correctness(self):
        self._test_correctness(leaky_cavity_8, Dopri5(), num_tsave=11)

    def test_td_correctness(self):
        self._test_correctness(damped_tdqubit, Dopri5(), num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_leaky_cavity_8, Dopri5(), Autograd(), num_tsave=11)

    def test_td_autograd(self):
        self._test_gradient(grad_damped_tdqubit, Dopri5(), Autograd(), num_tsave=11)
