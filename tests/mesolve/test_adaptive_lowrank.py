import jax
import pytest

import dynamiqs as dq
from dynamiqs import Options
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.method import LowRank, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, dia_ocavity, otdqubit

# we only test Tsit5 to keep the unit test suite fast


# use double precision for gradients
@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    # Keep precision changes local to this module to avoid cross-test leakage.
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')  # needed for time dependent test
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def _lowrank_method(system):
    M = 2 if system is otdqubit else system.n // 2
    return LowRank(M=M, ode_method=Tsit5())


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveLowRank(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    def test_correctness(self, system):
        options = Options()
        self._test_correctness(system, _lowrank_method(system), options=options)

    @pytest.mark.parametrize('system', [dense_ocavity, dia_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, _lowrank_method(system), gradient)
