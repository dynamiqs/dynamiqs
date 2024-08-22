import jax
import jax.numpy as jnp
import pytest

from dynamiqs.options import Options
from dynamiqs.solver import Tsit5

from ..integrator_tester import IntegratorTester
from .floquet_qubit import FloquetQubit, FloquetQubit_t


class TestFloquet(IntegratorTester):
    @pytest.mark.parametrize('omega', 2.0 * jnp.pi * jnp.array([1.0, 2.5]))
    @pytest.mark.parametrize('amp', 2.0 * jnp.pi * jnp.array([0.01, 0.1]))
    @pytest.mark.parametrize('omega_d_frac', [0.9, 0.9999])
    def test_correctness_0(self, omega, amp, omega_d_frac, ysave_atol: float = 1e-3):
        omega_d = omega_d_frac * omega
        floquet_qubit_0 = FloquetQubit(omega, omega_d, amp)
        floquet_result_0 = floquet_qubit_0.run(Tsit5())
        floquet_modes_0 = floquet_result_0.floquet_modes
        quasienergies = floquet_result_0.quasienergies
        true_floquet_modes_0 = floquet_qubit_0.state(0)
        true_quasienergies = floquet_qubit_0.true_quasienergies()
        # sort them appropriately for comparison
        idxs = jnp.argmin(
            jnp.abs(quasienergies - true_quasienergies[..., None]), axis=1
        )
        state_errs = jnp.linalg.norm(
            floquet_modes_0[:, idxs] - true_floquet_modes_0, axis=(0, 1)
        )
        assert jnp.all(state_errs <= ysave_atol)
        quasi_errs = jnp.linalg.norm(quasienergies[idxs] - true_quasienergies, axis=-1)
        assert jnp.all(quasi_errs <= ysave_atol)

    @pytest.mark.parametrize('omega', 2.0 * jnp.pi * jnp.array([2.5]))
    @pytest.mark.parametrize('amp', 2.0 * jnp.pi * jnp.array([0.01]))
    @pytest.mark.parametrize('omega_d_frac', [0.9])
    @pytest.mark.parametrize('t', [0.1, 0.3])
    @pytest.mark.parametrize('precompute_0_for_t', [True, False])
    @pytest.mark.parametrize('save_states', [True, False])
    def test_correctness_t(
        self,
        omega,
        amp,
        omega_d_frac,
        t,
        save_states,
        precompute_0_for_t,
        ysave_atol: float = 1e-3,
    ):
        omega_d = omega_d_frac * omega
        tsave = jnp.linspace(0.0, t, 4)
        if precompute_0_for_t:
            floquet_qubit_0 = FloquetQubit(omega, omega_d, amp)
            floquet_result_0 = floquet_qubit_0.run(Tsit5())
            floquet_modes_0 = floquet_result_0.floquet_modes
            quasienergies = floquet_result_0.quasienergies
        else:
            floquet_modes_0 = None
            quasienergies = None
        floquet_qubit = FloquetQubit_t(
            omega,
            omega_d,
            amp,
            tsave,
            floquet_modes_0=floquet_modes_0,
            quasienergies=quasienergies,
        )
        floquet_result = floquet_qubit.run(
            Tsit5(), options=Options(save_states=save_states)
        )
        floquet_modes = floquet_result.floquet_modes
        state_phases = jnp.angle(floquet_modes[..., 0, :])
        floquet_modes = jnp.einsum(
            '...ij,...j->...ij', floquet_modes, jnp.exp(-1j * state_phases)
        )
        quasienergies = floquet_result.quasienergies
        if save_states:
            true_floquet_modes = jax.vmap(floquet_qubit.state)(tsave)
        else:
            true_floquet_modes = floquet_qubit.state(t)
        true_quasienergies = floquet_qubit.true_quasienergies()
        # sort them appropriately for comparison
        idxs = jnp.argmin(
            jnp.abs(quasienergies - true_quasienergies[..., None]), axis=1
        )
        state_errs = jnp.linalg.norm(
            floquet_modes[:, idxs] - true_floquet_modes, axis=(-1, -2)
        )
        assert jnp.all(state_errs <= ysave_atol)
        quasi_errs = jnp.linalg.norm(quasienergies[idxs] - true_quasienergies, axis=-1)
        assert jnp.all(quasi_errs <= ysave_atol)
