import jax
import jax.numpy as jnp
import pytest

from dynamiqs.method import Tsit5
from dynamiqs.utils.general import fidelity

from ..order import TEST_LONG
from .floquet_qubit import FloquetQubit

EPS = 1e-6


@pytest.mark.run(order=TEST_LONG)
class TestFloquet:
    @pytest.mark.parametrize('omega', 2.0 * jnp.pi * jnp.array([1.0, 2.5]))
    @pytest.mark.parametrize('amp', 2.0 * jnp.pi * jnp.array([0.01, 0.1]))
    @pytest.mark.parametrize('t', [0.0])
    @pytest.mark.parametrize('omega_d_frac', [0.9, 0.9999])
    def test_correctness(self, omega, amp, t, omega_d_frac, ysave_atol: float = 1e-5):
        # temporary fix for https://github.com/patrick-kidger/diffrax/issues/488
        tsave = jnp.array([0.0]) if t == 0.0 else jnp.linspace(0.0, t, 4)
        omega_d = omega_d_frac * omega
        floquet_qubit = FloquetQubit(omega, omega_d, amp, tsave)
        floquet_result = floquet_qubit.run(Tsit5())
        quasienergies = floquet_result.quasienergies
        true_modes = jax.vmap(floquet_qubit.state)(tsave)
        true_quasienergies = floquet_qubit.true_quasienergies()
        # sort them appropriately for comparison
        idxs = jnp.argmin(
            jnp.abs(quasienergies - true_quasienergies[..., None]), axis=1
        )
        mode_infid = 1 - jnp.average(
            fidelity(true_modes, floquet_result.modes[:, idxs])
        )
        assert jnp.allclose(quasienergies[idxs], true_quasienergies, atol=ysave_atol)
        assert 0 - EPS <= mode_infid < ysave_atol + EPS
