from math import pi

import torchqdynamics as tq

from .closed_system import Cavity
from .test_sesolve import TestSESolve

cavity_8 = Cavity(n=8, delta=2 * pi, alpha0=1.0)


class TestSEEuler(TestSESolve):
    def test_batching(self):
        options = tq.options.Euler(dt=1e-2)
        self._test_batching(options, cavity_8)

    def test_psi_save(self):
        options = tq.options.Euler(dt=1e-4)
        self._test_psi_save(options, cavity_8, num_t_save=11)
