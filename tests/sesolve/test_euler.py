import pytest

import torchqdynamics as tq

from .closed_system import cavity_8
from .sesolver_test import SESolverTest


class TestSEEuler(SESolverTest):
    def test_batching(self):
        solver = tq.solver.Euler(dt=1e-2)
        self._test_batching(solver, cavity_8)

    @pytest.mark.skip(reason='failing - to fix')
    @pytest.mark.long
    def test_psi_save(self):
        solver = tq.solver.Euler(dt=1e-4)
        self._test_psi_save(solver, cavity_8, nt=51, rtol=1e-3, atol=1e-3)
