from dynamiqs.gradient import Autograd
from dynamiqs.solver import Rouchon1

from .monitored_system import gmcavity, mcavity
from .sme_solver_tester import SMESolverTester


class TestMERouchon1(SMESolverTester):
    def test_batching(self):
        solver = Rouchon1(dt=1e-2)
        self._test_batching(mcavity, solver)

    def test_correctness(self):
        solver = Rouchon1(dt=1e-3)
        self._test_correctness(
            mcavity, solver, ysave_atol=1e-2, esave_rtol=1e-4, esave_atol=1e-2
        )

    def test_autograd(self):
        solver = Rouchon1(dt=1e-3)
        self._test_gradient(gmcavity, solver, Autograd(), rtol=1e-4, atol=1e-2)
