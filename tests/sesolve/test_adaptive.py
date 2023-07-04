from math import pi

import dynamiqs as dq

from .closed_system import Cavity
from .se_solver_tester import SESolverTester

cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0)


class TestAdaptive(SESolverTester):
    def test_batching(self):
        options = dq.options.Dopri5()
        self._test_batching(options, cavity_8)

    def test_y_save(self):
        options = dq.options.Dopri5()
        self._test_y_save(options, cavity_8, num_t_save=11)
