import jax.numpy as jnp
import pytest

from dynamiqs.solver import Tsit5

from ..integrator_tester import IntegratorTester
from .floquet_qubit import FloquetQubit


class TestFloquet(IntegratorTester):
    @pytest.mark.parametrize('t', [0.0, 0.654])
    @pytest.mark.parametrize('omega_d_frac', [0.9999])
    def test_correctness(self, t, omega_d_frac, ysave_atol: float = 1e-4):
        amp = 2.0 * jnp.pi * 0.001
        omega = 2.0 * jnp.pi * 1.0
        omega_d = omega_d_frac * omega
        floquet_qubit = FloquetQubit(omega=omega, omega_d=omega_d, amp=amp, t_mode=t)
        floquet_result = floquet_qubit.run(Tsit5())
        floquet_modes = floquet_result.floquet_modes
        g_phases = jnp.angle(floquet_modes[:, 0, 0])
        floquet_modes_nophase = jnp.einsum(
            'i,ijd->ijd', jnp.exp(-1j * g_phases), floquet_modes
        )
        quasi_energies = floquet_result.quasi_energies
        true_floquet_modes = floquet_qubit.state(t)
        true_quasi_energies = floquet_qubit.quasi_energies()
        idxs = jnp.argmin(
            jnp.abs(quasi_energies - true_quasi_energies[..., None]), axis=1
        )
        state_errs = jnp.linalg.norm(
            floquet_modes_nophase[idxs] - true_floquet_modes, axis=(0, 1)
        )
        assert jnp.all(state_errs <= ysave_atol)
        quasi_errs = jnp.linalg.norm(
            quasi_energies[idxs] - true_quasi_energies, axis=-1
        )
        assert jnp.all(quasi_errs <= ysave_atol)
