import jax.numpy as jnp
import pytest

from dynamiqs.method import Tsit5
from dynamiqs.options import Options

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity_steady


@pytest.mark.run(order=TEST_LONG)
class TestMESolveSteadyState(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity_steady])
    def test_correctness(self, system):
        options = Options(steady_state=True)
        result = system.run(Tsit5(), options=options)
        states = result.states.to_jax()
        final_state = states[-1]
        assert jnp.allclose(
            final_state, system.steady_state().to_jax(), atol=1e-3, rtol=1e-3
        )
