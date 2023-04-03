import numpy as np
import pytest

import torchqdynamics as tq

from .closed_system import Cavity
from .sesolver_test import SESolverTest

cavity_8 = Cavity(n=8, delta=2 * np.pi, alpha0=1.0)


class TestSEEuler(SESolverTest):
    def test_batching(self):
        solver = tq.solver.Euler(dt=1e-2)
        self._test_batching(solver, cavity_8)

    @pytest.mark.long
    def test_psi_save(self):
        solver = tq.solver.Euler(dt=1e-5)
        self._test_psi_save(solver, cavity_8, num_t_save=51, rtol=1e-2)
