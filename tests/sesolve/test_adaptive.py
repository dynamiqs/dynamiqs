from math import pi

import torchqdynamics as tq

from .closed_system import Cavity
from .sesolver_tester import SESolverTester

cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0)


class TestAdaptive(SESolverTester):
    def test_batching(self):
        options = tq.options.Dopri45()
        self._test_batching(options, cavity_8)

    def test_rho_save(self):
        options = tq.options.Dopri45()
        self._test_psi_save(options, cavity_8, num_t_save=11)
