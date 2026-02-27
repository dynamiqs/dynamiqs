import time

import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG, TEST_SHORT


def rand_jssesolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 4
    kH, *kLs, kpsi0, kEs, kmc = jax.random.split(jax.random.key(31), nkeys)
    H = dq.random.herm(kH, (*nH, n, n))
    # H = dq.random.operator(kH, n, batch=nH)
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

@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('cartesian_batching', [True, False])
# @pytest.mark.parametrize('H_batch', [()])
# @pytest.mark.parametrize('Ls_batch', [[()], [(),()]])
# @pytest.mark.parametrize('psi0_batch', [()])
# @pytest.mark.parametrize('ntrajs', [1, 5])
@pytest.mark.parametrize('H_batch', [(), (3, 2)])
@pytest.mark.parametrize('Ls_batch', [[(), ()], [(2,), ()], [(3, 2), (2,)]])
@pytest.mark.parametrize('psi0_batch', [(), (2,), (3, 2)])
@pytest.mark.parametrize('ntrajs', [1, 2])
def test_keys_batching(cartesian_batching, H_batch, Ls_batch, psi0_batch, ntrajs):
    n = 2
    nEs = 6
    ntsave = 7

    # create base (unbatched) operators with identical data
    H = dq.random.herm(jax.random.key(0), (n, n))
    Ls = [dq.random.operator(jax.random.key(i + 1), n) for i in range(len(Ls_batch))]
    psi0 = dq.random.ket(jax.random.key(len(Ls_batch) + 1), n)
    Es = dq.random.operator(
        jax.random.key(len(Ls_batch) + 2), n, hermitian=False, batch=nEs
    )

    # broadcast to desired batch shapes (identical data along batch dims)
    H = H.broadcast_to(*H_batch, n, n)
    Ls = [L.broadcast_to(*lb, n, n) for L, lb in zip(Ls, Ls_batch)]
    psi0 = psi0.broadcast_to(*psi0_batch, n, 1)

    # sanity-check input shapes
    # assert H.shape == (*H_batch, n, n)
    # for L, lb in zip(Ls, Ls_batch):
    #     assert L.shape == (*lb, n, n)
    # assert psi0.shape == (*psi0_batch, n, 1)

    # verify inputs are identical along batch dimensions
    if H_batch:
        H_flat = H.to_jax().reshape(-1, n, n)
        assert jnp.allclose(H_flat[0], H_flat[-1]), (
            'H data differs across batch dims — test expects identical data'
        )
    for L, lb in zip(Ls, Ls_batch):
        if lb:
            L_flat = L.to_jax().reshape(-1, n, n)
            assert jnp.allclose(L_flat[0], L_flat[-1]), (
                'L data differs across batch dims — test expects identical data'
            )
    if psi0_batch:
        psi0_flat = psi0.to_jax().reshape(-1, n, 1)
        assert jnp.allclose(psi0_flat[0], psi0_flat[-1]), (
            'psi0 data differs across batch dims — test expects identical data'
        )

    keys = jax.random.split(jax.random.key(len(Ls_batch) + 3), num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1, ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.Event(dtmax=1e-1)

    result = dq.jssesolve(
        H, Ls, psi0, tsave, keys=keys, exp_ops=Es, options=options, method=method
    )

    # check expected output shape
    if cartesian_batching:
        batch_shape = (*H_batch, *(d for lb in Ls_batch for d in lb), *psi0_batch)
    else:
        batch_shape = jnp.broadcast_shapes(H_batch, *Ls_batch, psi0_batch)

    assert result.states.shape == (*batch_shape, ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*batch_shape, ntrajs, nEs, ntsave)

    # verify keys are independent across batch dims
    nbatch = len(batch_shape)
    if nbatch > 0:
        total_batch = 1
        for d in batch_shape:
            total_batch *= d

        if total_batch > 1:
            flat_keys = result.keys.reshape(total_batch, ntrajs, -1)
            # check multiple batch element pairs, not just first vs last
            for i in range(min(total_batch, 4)):
                for j in range(i + 1, min(total_batch, 4)):
                    assert not jnp.array_equal(flat_keys[i], flat_keys[j]), (
                        f'Keys identical for batch elements {i} and {j} '
                        '— key splitting is broken'
                    )
            # check trajectories within a batch element are distinct
            if ntrajs > 1:
                assert not jnp.array_equal(flat_keys[0, 0], flat_keys[0, 1]), (
                    'Trajectory keys within a batch element are identical'
                )
