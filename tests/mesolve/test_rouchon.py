import numpy as np
import pytest

import torchqdynamics as tq

from .mesolver_tester import MESolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * np.pi, delta=2 * np.pi, alpha0=1.0)


class TestMERouchon1(MESolverTester):
    def test_batching(self):
        options = tq.options.Rouchon1(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_rho_save(self):
        options = tq.options.Rouchon1(dt=1e-4)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)


class TestMERouchon1_5(MESolverTester):
    @pytest.mark.skip(reason='failing - to fix')
    def test_batching(self):
        options = tq.options.Rouchon1_5(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    @pytest.mark.skip(reason='failing - to fix')
    def test_rho_save(self):
        options = tq.options.Rouchon1_5(dt=1e-4)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)


class TestMERouchon2(MESolverTester):
    def test_batching(self):
        options = tq.options.Rouchon2(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_rho_save(self):
        options = tq.options.Rouchon2(dt=1e-3)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)
