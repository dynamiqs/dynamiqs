import jax
import jax.numpy as jnp
import pytest

from dynamiqs.method import Tsit5

from ..order import TEST_LONG
from ..systems import floquet_qubit


@pytest.mark.run(order=TEST_LONG)
class TestFloquet:
    def test_correctness(self, ysave_atol: float = 1e-5):
        method = Tsit5()
        tsave = floquet_qubit.tsave
        floquet_result = floquet_qubit.run(method)
        quasienergies = floquet_result.quasienergies
        true_modes = jax.vmap(floquet_qubit.state)(tsave).to_jax()
        true_quasienergies = floquet_qubit.true_quasienergies()
        # sort them appropriately for comparison
        idxs = jnp.argmin(
            jnp.abs(quasienergies - true_quasienergies[..., None]), axis=1
        )
        sorted_modes = floquet_result.modes[:, idxs].to_jax()
        assert jnp.allclose(quasienergies[idxs], true_quasienergies, atol=ysave_atol)
        assert jnp.allclose(true_modes, sorted_modes, atol=ysave_atol)
