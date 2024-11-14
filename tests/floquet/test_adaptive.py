import jax
import jax.numpy as jnp
import pytest

from dynamiqs.options import Options
from dynamiqs.solver import Tsit5

from .floquet_qubit import FloquetQubit


class TestFloquet:
    @pytest.mark.parametrize('omega', 2.0 * jnp.pi * jnp.array([1.0, 2.5]))
    @pytest.mark.parametrize('amp', 2.0 * jnp.pi * jnp.array([0.01, 0.1]))
    @pytest.mark.parametrize('t', [0.0, 0.1, 0.3])
    @pytest.mark.parametrize('omega_d_frac', [0.9, 0.9999])
    @pytest.mark.parametrize('save_states', [True, False])
    def test_correctness(
        self, omega, amp, t, omega_d_frac, save_states, ysave_atol: float = 1e-3
    ):
        # temporary fix for https://github.com/patrick-kidger/diffrax/issues/488
        tsave = jnp.array([0.0]) if t == 0.0 else jnp.linspace(0.0, t, 4)
        omega_d = omega_d_frac * omega
        floquet_qubit = FloquetQubit(omega, omega_d, amp, tsave)
        floquet_result = floquet_qubit.run(
            Tsit5(), options=Options(save_states=save_states)
        )
        modes = floquet_result.modes
        state_phases = jnp.angle(modes[..., :, 0, 0])
        modes = jnp.einsum(
            '...ijk,...i->...ijk', modes, jnp.exp(-1j * state_phases)
        )
        quasienergies = floquet_result.quasienergies
        if save_states:
            true_modes = jax.vmap(floquet_qubit.state)(tsave)
        else:
            true_modes = floquet_qubit.state(t)
        true_quasienergies = floquet_qubit.true_quasienergies()
        # sort them appropriately for comparison
        idxs = jnp.argmin(
            jnp.abs(quasienergies - true_quasienergies[..., None]), axis=1
        )
        state_errs = jnp.linalg.norm(
            modes[:, idxs] - true_modes, axis=(-1, -2)
        )
        assert jnp.all(state_errs <= ysave_atol)
        quasi_errs = jnp.linalg.norm(quasienergies[idxs] - true_quasienergies, axis=-1)
        assert jnp.all(quasi_errs <= ysave_atol)
