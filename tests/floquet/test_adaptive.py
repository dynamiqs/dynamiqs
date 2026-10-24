import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import block_diag

import dynamiqs as dq
from dynamiqs.integrators.core.floquet_integrator import _diagonalize_unitary
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

    def test_degenerate_modes_are_orthonormal(self):
        # resonant ladder: quasienergies cluster at 0 and at the zone edge π/T
        n, T = 4, 1.0
        key_1, key_2 = jax.random.split(jax.random.PRNGKey(42))
        _, R = jnp.linalg.eigh(dq.random.herm(key_1, (n, n)).to_jax())
        H0 = R @ jnp.diag(jnp.pi / T * jnp.arange(n)) @ R.mT.conj()
        V = dq.random.herm(key_2, (n, n))
        H = dq.constant(H0) + dq.modulated(
            lambda t: 1e-3 * jnp.cos(2 * jnp.pi * t / T), V
        )

        result = dq.floquet(H, T, jnp.array([0.0]))
        modes = result.modes.to_jax()[0, ..., 0].T
        U = dq.sepropagator(H, [0.0, T]).final_propagator.to_jax()
        evals = jnp.exp(-1j * result.quasienergies * T)
        assert jnp.allclose(modes.mT.conj() @ modes, jnp.eye(n), atol=1e-5)
        assert jnp.allclose(U @ modes, modes * evals, atol=1e-5)

    def test_eigenvalue_at_minus_one(self):
        # sigmax has the exact eigenvalues ±1, the other block a near-degenerate cluster
        _, R = jnp.linalg.eigh(dq.random.herm(jax.random.PRNGKey(42), (3, 3)).to_jax())
        cluster = jnp.exp(1j * jnp.array([2.0, 2.0 + 1e-4, 2.0 + 2e-4]))
        U = block_diag(dq.sigmax().to_jax(), (R * cluster) @ R.mT.conj())

        evals, evecs = _diagonalize_unitary(U)
        assert jnp.allclose(evecs.mT.conj() @ evecs, jnp.eye(5), atol=1e-5)
        assert jnp.allclose(U @ evecs, evecs * evals, atol=1e-5)
