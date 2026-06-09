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


@pytest.mark.run(order=TEST_INSTANT)
def test_vacuum():
    dim = 4

    # vacuum is the zero-photon Fock state
    assert np.allclose(dq.vacuum(dim), dq.fock(dim, 0))

    # check that no error is raised while tracing the function
    jax.jit(dq.vacuum, static_argnums=(0,)).trace(dim)


@pytest.mark.run(order=TEST_INSTANT)
def test_vacuum_dm():
    dim = 4

    assert np.allclose(dq.vacuum_dm(dim), dq.fock_dm(dim, 0))

    # check that no error is raised while tracing the function
    jax.jit(dq.vacuum_dm, static_argnums=(0,)).trace(dim)


@pytest.mark.run(order=TEST_INSTANT)
def test_cat():
    dim = 16
    alpha = 2.0
    alphas = jnp.array([1.0, 2.0, 3.0])
    thetas = jnp.array([0.0, jnp.pi])

    # even cat (theta=0) only populates even Fock states, odd cat (theta=pi) odd ones
    even_cat = dq.cat(dim, alpha)
    odd_cat = dq.cat(dim, alpha, jnp.pi)
    assert np.allclose(even_cat.to_jax()[1::2], 0.0, atol=1e-6)
    assert np.allclose(odd_cat.to_jax()[::2], 0.0, atol=1e-6)

    # default theta is the even cat
    assert np.allclose(dq.cat(dim, alpha), dq.cat(dim, alpha, 0.0))

    # cat states are normalized
    assert np.allclose(dq.norm(even_cat), 1.0)
    assert np.allclose(dq.norm(odd_cat), 1.0)

    # edge case: in the vanishing-amplitude limit the even cat is the vacuum |0>
    # and the odd cat is the single-photon Fock state |1>
    assert np.allclose(dq.cat(dim, 0.0), dq.fock(dim, 0))
    assert np.allclose(dq.cat(dim, 0.0, jnp.pi), dq.fock(dim, 1))
    # batching over alpha still resolves the limit for the zero entry
    zero_batch = dq.cat(dim, jnp.array([0.0, 2.0]), jnp.pi)
    assert np.allclose(zero_batch[0], dq.fock(dim, 1))
    assert np.allclose(dq.norm(zero_batch), 1.0)

    # batching over alpha matches the manually stacked result
    state1 = dq.cat(dim, alphas)
    state2 = dq.stack([dq.cat(dim, a) for a in alphas])
    assert np.allclose(state1, state2)
    assert state1.shape == (3, dim, 1)

    # alpha and theta are broadcast together
    assert dq.cat(dim, alphas[None, :], thetas[:, None]).shape == (2, 3, dim, 1)

    # check that no error is raised while tracing the function
    jax.jit(dq.cat, static_argnums=(0,)).trace(dim, alpha)
    jax.jit(dq.cat, static_argnums=(0,)).trace(dim, alphas)
    jax.jit(dq.cat, static_argnums=(0,)).trace(dim, alphas[None, :], thetas[:, None])


@pytest.mark.run(order=TEST_INSTANT)
def test_cat_dm():
    dim = 16
    alpha = 2.0
    alphas = jnp.array([1.0, 2.0, 3.0])
    thetas = jnp.array([0.0, jnp.pi])

    # cat_dm is the density matrix of cat
    assert np.allclose(dq.cat_dm(dim, alpha), dq.cat(dim, alpha).todm())

    # batching is preserved
    assert dq.cat_dm(dim, alphas[None, :], thetas[:, None]).shape == (2, 3, dim, dim)

    # check that no error is raised while tracing the function
    jax.jit(dq.cat_dm, static_argnums=(0,)).trace(dim, alpha)
    jax.jit(dq.cat_dm, static_argnums=(0,)).trace(dim, alphas)
    jax.jit(dq.cat_dm, static_argnums=(0,)).trace(dim, alphas[None, :], thetas[:, None])
