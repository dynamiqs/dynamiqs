import pytest

from dynamiqs.gradient import Autograd

from ..solver_tester import SolverTester
from .closed_system import cavity

Propagator = None


@pytest.mark.skip(reason='Propagator not implemented yet')
class TestSEPropagator(SolverTester):
    def test_batching(self):
        self._test_batching(cavity, Propagator())

    def test_correctness(self):
        self._test_correctness(cavity, Propagator())

    def test_autograd(self):
        self._test_gradient(cavity, Propagator(), Autograd())
