from math import pi

import pytest

import dynamiqs as dq

from .me_solver_tester import MEAdjointSolverTester
from .open_system import LeakyCavity

leaky_cavity_8 = LeakyCavity(n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0)
grad_leaky_cavity_8 = LeakyCavity(
    n=8, kappa=2 * pi, delta=2 * pi, alpha0=1.0, requires_grad=True
)


class TestMERouchon1(MEAdjointSolverTester):
    def test_batching(self):
        options = dq.options.Rouchon1(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_rho_save(self):
        options = dq.options.Rouchon1(dt=1e-4)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)

    def test_adjoint(self):
        options = dq.options.Rouchon1(dt=1e-3)
        self._test_adjoint(options, grad_leaky_cavity_8, num_t_save=11)

    def test_adjoint_expops(self):
        options = dq.options.Rouchon1(dt=1e-3)
        self._test_adjoint_expops(options, grad_leaky_cavity_8, num_t_save=11)


class TestMERouchon1_5(MEAdjointSolverTester):
    @pytest.mark.skip(reason='failing - to fix')
    def test_batching(self):
        options = dq.options.Rouchon1_5(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    @pytest.mark.skip(reason='failing - to fix')
    def test_rho_save(self):
        options = dq.options.Rouchon1_5(dt=1e-4)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)


class TestMERouchon2(MEAdjointSolverTester):
    def test_batching(self):
        options = dq.options.Rouchon2(dt=1e-2)
        self._test_batching(options, leaky_cavity_8)

    def test_rho_save(self):
        options = dq.options.Rouchon2(dt=1e-3)
        self._test_rho_save(options, leaky_cavity_8, num_t_save=11)

    def test_adjoint(self):
        options = dq.options.Rouchon2(dt=1e-3)
        self._test_adjoint(options, grad_leaky_cavity_8, num_t_save=11)

    def test_adjoint_expops(self):
        options = dq.options.Rouchon1(dt=1e-3)
        self._test_adjoint_expops(options, grad_leaky_cavity_8, num_t_save=11)
