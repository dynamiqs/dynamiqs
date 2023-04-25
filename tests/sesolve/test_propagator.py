import numpy as np

import torchqdynamics as tq

from .closed_system import Cavity
from .test_sesolve import SESolverTester

cavity_8 = Cavity(n=8, delta=2 * np.pi, alpha0=1.0)


class TestPropagator(SESolverTester):
    def test_batching(self):
        options = tq.options.Propagator()
        self._test_batching(options, cavity_8)

    def test_psi_save(self):
        options = tq.options.Propagator()
        self._test_psi_save(options, cavity_8, num_t_save=11)
