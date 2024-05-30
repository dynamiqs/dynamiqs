import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.solver import Tsit5

from ..solver_tester import SolverTester
from .closed_system import cavity, sparse_cavity, sparse_tdqubit, tdqubit

# we only test Tsit5 to keep the unit test suite fast


class TestSEAdaptive(SolverTester):
    @pytest.mark.parametrize('system', [cavity, sparse_cavity, tdqubit, sparse_tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [cavity, sparse_cavity, tdqubit, sparse_tdqubit])
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)
