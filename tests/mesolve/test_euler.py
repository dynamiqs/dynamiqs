from math import pi

import dynamiqs as dq

from .me_solver_tester import MESolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0)


class TestMEEuler(MESolverTester):
    def test_batching(self):
        options = dq.options.Euler(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_rho_save(self):
        options = dq.options.Euler(dt=1e-4)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)
