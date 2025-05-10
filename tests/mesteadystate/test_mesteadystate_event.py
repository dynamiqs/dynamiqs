import jax.numpy as jnp
import pytest

from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity_steady


@pytest.mark.run(order=TEST_LONG)
class TestSteadyState(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity_steady])
    def test_correctness(self, system):
        result = system.run(Tsit5())
        states = result.states.to_jax()
        final_state = states[-1]
        assert jnp.allclose(
            final_state, system.steady_state().to_jax(), atol=1e-3, rtol=1e-3
        )
