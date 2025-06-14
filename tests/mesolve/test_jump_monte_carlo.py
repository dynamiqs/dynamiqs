import jax
import pytest

from dynamiqs.method import JumpMonteCarlo

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import otdqubit


@pytest.mark.run(order=TEST_LONG)
class TestMESolveRouchon1(IntegratorTester):
    def test_correctness(self):
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 10_000)
        method = JumpMonteCarlo(keys)
        self._test_correctness(
            otdqubit, method, ysave_atol=1e-1, esave_rtol=1e-1, esave_atol=1e-1
        )
