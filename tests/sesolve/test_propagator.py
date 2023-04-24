import numpy as np

import torchqdynamics as tq

from .closed_system import Cavity
from .sesolver_test import SESolverTest

cavity_8 = Cavity(n=8, delta=2 * np.pi, alpha0=1.0)


class TestPropagator(SESolverTest):
    def test_batching(self):
        solver = tq.solver.Propagator()
        self._test_batching(solver, cavity_8)

    def test_psi_save(self):
        solver = tq.solver.Propagator()
        self._test_psi_save(solver, cavity_8, num_t_save=11)
