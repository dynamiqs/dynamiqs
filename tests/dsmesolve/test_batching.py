import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_SHORT

# ── key batching test ────────────────────────────────────────────────────────

@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('cartesian_batching', [True, False])
@pytest.mark.parametrize('H_batch, rho0_batch', [
    ((), ()),       # no batch
    ((2,), ()),     # H batched only
    ((), (2,)),     # rho0 batched only
    ((2,), (3,)),   # both batched (cartesian only, broadcast for flat)
])
@pytest.mark.parametrize('ntrajs', [1, 3])
def test_keys_batching(cartesian_batching, H_batch, rho0_batch, ntrajs):
    n = 2
    ntsave = 3

    # skip incompatible cartesian=False with non-broadcastable shapes
    if not cartesian_batching and H_batch and rho0_batch:
        try:
            jnp.broadcast_shapes(H_batch, rho0_batch)
        except ValueError:
            pytest.skip('non-broadcastable shapes for flat batching')

    H = dq.random.herm(jax.random.key(0), (*H_batch, n, n))
    jump_ops = [dq.destroy(n)]
    etas = jnp.ones(1)
    rho0 = dq.fock_dm(n, 1)
    if rho0_batch:
        rho0 = rho0.broadcast_to(*rho0_batch, n, n)

    keys = jax.random.split(jax.random.key(123), num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1 * (ntsave - 1), ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.EulerMaruyama(dt=1e-1)

    result = dq.dsmesolve(
        H, jump_ops, etas, rho0, tsave, keys=keys,
        options=options, method=method,
    )

    # expected batch shape
    if cartesian_batching:
        batch_shape = (*H_batch, *rho0_batch)
    else:
        batch_shape = jnp.broadcast_shapes(H_batch, rho0_batch)

    assert result.keys.shape == (*batch_shape, ntrajs)

    # key independence across batch elements
    total_batch = 1
    for d in batch_shape:
        total_batch *= d

    if total_batch > 1:
        flat_keys = result.keys.reshape(total_batch, ntrajs, -1)
        for i in range(min(total_batch, 4)):
            for j in range(i + 1, min(total_batch, 4)):
                assert not jnp.array_equal(flat_keys[i], flat_keys[j]), (
                    f'Keys identical for batch elements {i} and {j}'
                )
        if ntrajs > 1:
            assert not jnp.array_equal(flat_keys[0, 0], flat_keys[0, 1]), (
                'Trajectory keys within a batch element are identical'
            )


# ── trajectory independence test ─────────────────────────────────────────────

@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_keys_trajectory_independence(cartesian_batching):
    n = 8
    a = dq.destroy(n)

    H = (a.dag() @ a + 0.5 * (a + a.dag())).broadcast_to(2, n, n)
    jump_ops = [a]
    etas = jnp.ones(1)
    rho0 = dq.fock_dm(n, 5)

    ntrajs = 3
    ntsave = 11
    keys = jax.random.split(jax.random.key(42), ntrajs)
    tsave = jnp.linspace(0.0, 1.0, ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.EulerMaruyama(dt=1e-1)

    result = dq.dsmesolve(
        H, jump_ops, etas, rho0, tsave, keys=keys,
        options=options, method=method,
    )

    assert result.keys.shape == (2, ntrajs)

    final = result.final_state.to_jax()
    batch0 = final[0].reshape(ntrajs, -1)
    batch1 = final[1].reshape(ntrajs, -1)
    n_different = sum(
        1 for t in range(ntrajs)
        if not jnp.allclose(batch0[t], batch1[t], atol=1e-6)
    )
    assert n_different > 0, (
        'All trajectory pairs across batch elements have identical final states — '
        'key independence is broken'
    )
