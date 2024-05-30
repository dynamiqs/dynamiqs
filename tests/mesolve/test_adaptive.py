import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd
from dynamiqs.solver import Tsit5

from ..solver_tester import SolverTester
from .open_system import ocavity, otdqubit, sparse_ocavity, sparse_otdqubit

# we only test Tsit5 to keep the unit test suite fast


class TestMEAdaptive(SolverTester):
    @pytest.mark.parametrize(
        'system', [ocavity, sparse_ocavity, otdqubit, sparse_otdqubit]
    )
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize(
        'system', [ocavity, sparse_ocavity, otdqubit, sparse_otdqubit]
    )
    @pytest.mark.parametrize('gradient', [Autograd(), CheckpointAutograd()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)
