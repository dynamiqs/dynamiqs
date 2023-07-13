from math import pi

import dynamiqs as dq

from .me_solver_tester import MESolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0)


class TestAdaptive(MESolverTester):
    def test_batching(self):
        options = dq.options.Dopri5()
        self._test_batching(options, leaky_cavity_8)

    def test_correctness(self):
        options = dq.options.Dopri5()
        self._test_correctness(options, leaky_cavity_8, num_t_save=11)
