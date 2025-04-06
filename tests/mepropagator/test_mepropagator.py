import jax.numpy as jnp
import pytest

from dynamiqs import dag, mepropagator, operator_to_vector, vector_to_operator
from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..mesolve.open_system import dense_ocavity, otdqubit
from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
class TestMEPropagator(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, system, ysave_atol: float = 1e-4):
        params = system.params_default
        H = system.H(params)
        Ls = system.Ls(params)
        y0 = system.y0(params)
        rho0 = y0 @ dag(y0)
        propresult = mepropagator(H, Ls, system.tsave, method=Tsit5())
        true_ysave = system.states(system.tsave).to_jax()
        prop_ysave = (
            vector_to_operator(propresult.propagators @ operator_to_vector(rho0))
        ).to_jax()
        assert jnp.allclose(true_ysave, prop_ysave, atol=ysave_atol)
