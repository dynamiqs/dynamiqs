import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_coherent():
    alpha1, alpha2 = 1.0, 1.0j
    alphas1, alphas2 = np.linspace(0, 1, 5), 1j * np.linspace(0, 1, 7)[:, None]
    n1, n2 = 8, 8

    # short tensor product
    state1 = dq.coherent(n1, alpha1) & dq.coherent(n2, alpha2)
    state2 = dq.coherent((n1, n2), (alpha1, alpha2))
    assert np.allclose(state1, state2)

    # short batching
    state1 = dq.coherent(n1, alphas1)
    state2 = dq.stack([dq.coherent(n1, alpha) for alpha in alphas1])
    assert np.allclose(state1, state2)

    # short batching + tensor product
    state1 = dq.coherent(n1, alphas1) & dq.coherent(n2, alpha2)
    state2 = dq.coherent((n1, n2), (alphas1, alpha2))
    assert np.allclose(state1, state2)

    # double short batching + tensor product
    state1 = dq.coherent(n1, alphas1) & dq.coherent(n2, alphas2)
    state2 = dq.coherent((n1, n2), (alphas1, alphas2))
    assert np.allclose(state1, state2)

    # double short batching + tensor product with single full qarray
    state1 = dq.coherent(n1, alphas1) & dq.coherent(n2, alphas2)
    state2 = dq.coherent((n1, n2), (alphas1, alphas2))
    assert np.allclose(state1, state2)

    # check that no error is raised while tracing the function
    jax.jit(dq.coherent, static_argnums=(0,)).trace(n1, alpha1)
    jax.jit(dq.coherent, static_argnums=(0,)).trace(n1, alphas1)
    jax.jit(dq.coherent, static_argnums=(0,)).trace((n1, n2), (alpha1, alpha2))
    jax.jit(dq.coherent, static_argnums=(0,)).trace((n1, n2), (alphas1, alphas2))


@pytest.mark.run(order=TEST_INSTANT)
def test_coherent_dm():
    # prepare inputs
    alpha1, alpha2 = 1.0, 1.0j
    alphas1, alphas2 = np.linspace(0, 1, 5), 1j * np.linspace(0, 1, 7)[:, None]
    n1, n2 = 8, 8

    # check that no error is raised while tracing the function
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(n1, alpha1)
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace(n1, alphas1)
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace((n1, n2), (alpha1, alpha2))
    jax.jit(dq.coherent_dm, static_argnums=(0,)).trace((n1, n2), (alphas1, alphas2))


@pytest.mark.run(order=TEST_INSTANT)
def test_fock():
    # prepare inputs
    dim = 4
    dims = (4, 4)
    num = 1
    nums = jnp.array([1, 2, 3])

    # check that no error is raised while tracing the function
    jax.jit(dq.fock, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.fock, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.fock, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.fock, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)


@pytest.mark.run(order=TEST_INSTANT)
def test_fock_dm():
    # prepare inputs
    dim = 4
    dims = (4, 4)
    num = 1
    nums = jnp.array([1, 2, 3])

    # check that no error is raised while tracing the function
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.fock_dm, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)


@pytest.mark.run(order=TEST_INSTANT)
def test_basis():
    # prepare inputs
    dim = 4
    dims = (4, 4)
    num = 1
    nums = jnp.array([1, 2, 3])

    # check that no error is raised while tracing the function
    jax.jit(dq.basis, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.basis, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.basis, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.basis, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)


@pytest.mark.run(order=TEST_INSTANT)
def test_basis_dm():
    # prepare inputs
    dim = 4
    dims = (4, 4)
    num = 1
    nums = jnp.array([1, 2, 3])

    # check that no error is raised while tracing the function
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dim, num)
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dim, nums)
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dims, (num, num))
    jax.jit(dq.basis_dm, static_argnums=(0,)).trace(dims, jnp.stack([nums, nums]).T)


@pytest.mark.run(order=TEST_INSTANT)
def test_thermal_dm():
    # prepare inputs
    dim = 4
    dims = (4, 4)
    nth = 0.1
    nths = jnp.array([0.1, 0.2, 0.3])

    # check that no error is raised while tracing the function
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dim, nth)
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dim, nths)
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dims, (nth, nth))
    jax.jit(dq.thermal_dm, static_argnums=(0,)).trace(dims, jnp.stack([nths, nths]).T)


@pytest.mark.run(order=TEST_INSTANT)
def test_ground():
    # check that no error is raised while tracing the function
    jax.jit(dq.ground).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_ground_dm():
    # check that no error is raised while tracing the function
    jax.jit(dq.ground_dm).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_excited():
    # check that no error is raised while tracing the function
    jax.jit(dq.excited).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_excited_dm():
    # check that no error is raised while tracing the function
    jax.jit(dq.excited_dm).trace()
