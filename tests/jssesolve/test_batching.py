import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG, TEST_SHORT


def rand_jssesolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 4
    kH, *kLs, kpsi0, kEs, kmc = jax.random.split(jax.random.key(31), nkeys)
    H = dq.random.herm(kH, (*nH, n, n))
    Ls = [dq.random.operator(kL, n, batch=nL) for kL, nL in zip(kLs, nLs, strict=False)]
    psi0 = dq.random.ket(kpsi0, n, batch=npsi0)
    Es = dq.random.operator(kEs, n, hermitian=False, batch=nEs)
    return H, Ls, psi0, Es, kmc


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
@pytest.mark.parametrize('nL', [(), (6,)])
def test_cartesian_batching(nH, npsi0, nL):
    n = 2
    nLs = [nL]
    ntrajs = 7
    nEs = 8
    ntsave = 9

    # run jssesolve
    H, Ls, psi0, Es, kmc = rand_jssesolve_args(n, nH, nLs, npsi0, nEs)
    keys = jax.random.split(kmc, num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1, ntsave)
    method = dq.method.Event(dtmax=1e-1)
    result = dq.jssesolve(H, Ls, psi0, tsave, keys=keys, exp_ops=Es, method=method)

    # check result shape
    assert result.states.shape == (*nH, *nL, *npsi0, ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*nH, *nL, *npsi0, ntrajs, nEs, ntsave)
    assert result.keys.shape == (*nH, *nL, *npsi0, ntrajs)


# H has fixed shape (3, 4, n, n) for the next test case, we test a broad ensemble of
# compatible broadcastable shape
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nL1', [(), (4, 1, 3)])
@pytest.mark.parametrize('npsi0', [(), (1,), (3,), (2, 1), (2, 3), (4, 1, 3)])
@pytest.mark.parametrize('ntrajs', [1, 5])
def test_flat_batching(nL1, npsi0, ntrajs):
    n = 2
    nH = (2, 3)
    nLs = [nL1, ()]
    nEs = 7
    ntsave = 8

    # run jssesolve
    H, Ls, psi0, Es, kmc = rand_jssesolve_args(n, nH, nLs, npsi0, nEs)
    keys = jax.random.split(kmc, num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1, ntsave)
    options = dq.Options(cartesian_batching=False)
    method = dq.method.Event(dtmax=1e-1)
    result = dq.jssesolve(
        H, Ls, psi0, tsave, keys=keys, exp_ops=Es, options=options, method=method
    )

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1, npsi0)
    assert result.states.shape == (*broadcast_shape, ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*broadcast_shape, ntrajs, nEs, ntsave)
    assert result.keys.shape == (*broadcast_shape, ntrajs)


# ── key batching test ────────────────────────────────────────────────────────


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('cartesian_batching', [True, False])
@pytest.mark.parametrize(
    ('H_batch', 'psi0_batch'),
    [
        ((), ()),  # no batch
        ((2,), ()),  # H batched only
        ((), (2,)),  # psi0 batched only
        ((2,), (3,)),  # both batched (cartesian only, broadcast for flat)
    ],
)
@pytest.mark.parametrize('ntrajs', [1, 3])
def test_keys_batching(cartesian_batching, H_batch, psi0_batch, ntrajs):
    n = 2
    ntsave = 3

    # skip incompatible cartesian=False with non-broadcastable shapes
    if not cartesian_batching and H_batch and psi0_batch:
        try:
            jnp.broadcast_shapes(H_batch, psi0_batch)
        except ValueError:
            pytest.skip('non-broadcastable shapes for flat batching')

    H = dq.random.herm(jax.random.key(0), (*H_batch, n, n))
    jump_ops = [dq.destroy(n)]
    psi0 = dq.fock(n, 1)
    if psi0_batch:
        psi0 = psi0.broadcast_to(*psi0_batch, n, 1)

    keys = jax.random.split(jax.random.key(123), num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1 * (ntsave - 1), ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.EulerJump(dt=1e-1)

    result = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, options=options, method=method
    )

    # expected batch shape
    if cartesian_batching:
        batch_shape = (*H_batch, *psi0_batch)
    else:
        batch_shape = jnp.broadcast_shapes(H_batch, psi0_batch)

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
    psi0 = dq.fock(n, 5)

    ntrajs = 3
    ntsave = 11
    keys = jax.random.split(jax.random.key(42), ntrajs)
    tsave = jnp.linspace(0.0, 1.0, ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.EulerJump(dt=1e-1)

    result = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, options=options, method=method
    )

    assert result.keys.shape == (2, ntrajs)

    final = result.final_state.to_jax()
    batch0 = final[0].reshape(ntrajs, -1)
    batch1 = final[1].reshape(ntrajs, -1)
    n_different = sum(
        1 for t in range(ntrajs) if not jnp.allclose(batch0[t], batch1[t], atol=1e-6)
    )
    assert n_different > 0, (
        'All trajectory pairs across batch elements have identical final states — '
        'key independence is broken'
    )
