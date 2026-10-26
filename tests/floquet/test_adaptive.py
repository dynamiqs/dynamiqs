import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core.floquet_integrator import _solve_by_blocks
from dynamiqs.method import Tsit5

from ..order import TEST_LONG, TEST_SHORT
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


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('ncols', [1, 8, 13, 32])
def test_solve_by_blocks(ncols):
    # a batch of two systems, with a number of columns that may not fill the last block
    def normal(key, ncols):
        return jax.random.normal(key, (2, 16, ncols), dtype=complex)

    keys = jax.random.split(jax.random.PRNGKey(42), 5)
    A = jnp.eye(16) + 0.1 * normal(keys[0], 16)
    dA = normal(keys[1], 16)
    B, dB, dX_bar = (normal(key, ncols) for key in keys[2:])

    # forward mode: X = A⁻¹B and dX = A⁻¹(dB - dA X)
    X, dX = jax.jvp(_solve_by_blocks, (A, B), (dA, dB))
    assert jnp.allclose(A @ X, B, atol=1e-5)
    assert jnp.allclose(A @ dX, dB - dA @ X, atol=1e-5)

    # reverse mode: the transpose of the forward derivative
    _, vjp = jax.vjp(_solve_by_blocks, A, B)
    dA_bar, dB_bar = vjp(dX_bar)
    lhs = jnp.sum(dX_bar * dX)
    rhs = jnp.sum(dA_bar * dA) + jnp.sum(dB_bar * dB)
    assert jnp.allclose(lhs, rhs, rtol=1e-4)
