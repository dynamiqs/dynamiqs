import numpy as np
import pytest

import torchqdynamics as tq

from .mesolver_test import MESolverTest
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=1.0, delta=2 * np.pi, alpha0=1.0)


class TestMEEuler(MESolverTest):
    def test_batching(self):
        solver = tq.solver.Euler(dt=1e-2)
        self._test_batching(solver, leaky_cavity_8)

    @pytest.mark.long
    def test_rho_save(self):
        solver = tq.solver.Euler(dt=1e-5)
        self._test_rho_save(solver, leaky_cavity_8, num_t_save=51, rtol=1e0)
