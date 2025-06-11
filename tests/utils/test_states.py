import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


# test for the `dq.coherent` method
@pytest.mark.run(order=TEST_INSTANT)
def test_coherent():
    alpha1, alpha2 = 1.0, 1.0j
    alphas1, alphas2 = np.linspace(0, 1, 5), np.linspace(0, 1, 7) * 1j
    n1, n2 = 8, 8

    # Short tensor product
    state1 = dq.coherent(n1, alpha1) & dq.coherent(n2, alpha2)
    state2 = dq.coherent((n1, n2), (alpha1, alpha2))
    assert np.allclose(state1, state2)

    # Short batching
    state1 = dq.coherent(n1, alphas1)
    state2 = dq.stack([dq.coherent(n1, alpha) for alpha in alphas1])
    assert np.allclose(state1, state2)

    # Short batching + tensor product
    state1 = dq.coherent(n1, alphas1) & dq.coherent(n2, alpha2)[None, ...]
    state2 = dq.coherent((n1, n2), (alphas1, alpha2))
    assert np.allclose(state1, state2)

    # Double short batching + tensor product
    state1 = (
        dq.coherent(n1, alphas1)[None, ...] & dq.coherent(n2, alphas2)[:, None, ...]
    )
    state2 = dq.coherent((n1, n2), (alphas1, alphas2[:, None]))
    assert np.allclose(state1, state2)

    # Double short batching + tensor product with single full qarray
    state1 = (
        dq.coherent(n1, alphas1)[None, ...] & dq.coherent(n2, alphas2)[:, None, ...]
    )
    state2 = dq.coherent((n1, n2), (alphas1[None, ...], alphas2[:, None]))
    assert np.allclose(state1, state2)


@pytest.mark.run(order=TEST_INSTANT)
def test_tracing():
    # prepare random keys and dimensions
    dim = 4
    dims = (4, 4)
    alpha = 0.5
    alphas = jnp.array([0.5, 1.0, 1.5])
    nth = 0.1
    nths = jnp.array([0.1, 0.2, 0.3])
    num = 1
    nums = jnp.array([1, 2, 3])

    # check that no error is raised while tracing the functions
    jax.jit(dq.fock, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.fock, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.fock, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.fock, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)
    jax.jit(dq.basis, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.basis, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.basis, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.basis, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)
    jax.jit(dq.coherent, static_argnums=(0,)).trace(dim, alpha)
    jax.jit(dq.coherent, static_argnums=(0,)).trace(dim, alphas)
    jax.jit(dq.coherent, static_argnums=(0,)).trace(dims, (alpha, alpha))
    jax.jit(dq.coherent, static_argnums=(0,)).trace(dims, jnp.stack([alphas, alphas]))
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(dim, alpha)
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(dim, alphas)
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(dims, (alpha, alpha))
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(
        dims, jnp.stack([alphas, alphas])
    )
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dim, nth)
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dim, nths)
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dims, (nth, nth))
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dims, jnp.stack([nths, nths]).T)
    jax.jit(dq.ground).trace()
    jax.jit(dq.ground_dm).trace()
    jax.jit(dq.excited).trace()
    jax.jit(dq.excited_dm).trace()
