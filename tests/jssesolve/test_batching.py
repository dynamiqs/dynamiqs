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


# ---------------------------------------------------------------------------
# Key-independence test: cheap simulation, wide variety of batch patterns
# ---------------------------------------------------------------------------
# All batch dims use size 2 for cheapness.  ntrajs uses 1 or 3 (never 2) so
# shape assertions stay unambiguous despite uniform batch sizes.
# Simulation time is near-zero — only key independence is tested here.
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('cartesian_batching', [True, False])
@pytest.mark.parametrize('H_batch', [(), (2,)])
@pytest.mark.parametrize('Ls_batch', [[(), ()], [(2, 2), (2,)]])
@pytest.mark.parametrize('psi0_batch', [()])
@pytest.mark.parametrize('ntrajs', [1, 3])
def test_keys_batching(cartesian_batching, H_batch, Ls_batch, psi0_batch, ntrajs):
    n = 2
    ntsave = 4
    nEs = 5  # distinct from n=2, ntrajs∈{1,3}, batch=2, ntsave=4

    H, Ls, psi0, Es, _ = rand_jssesolve_args(n, H_batch, Ls_batch, psi0_batch, nEs)
    keys = jax.random.split(jax.random.key(123), num=ntrajs)
    tsave = jnp.linspace(0.0, 0.001, ntsave)  # near-zero: only key shapes matter
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.Event(dtmax=1e-1)

    result = dq.jssesolve(
        H, Ls, psi0, tsave, keys=keys, exp_ops=Es, options=options, method=method
    )

    # expected batch shape
    if cartesian_batching:
        batch_shape = (*H_batch, *(d for lb in Ls_batch for d in lb), *psi0_batch)
    else:
        batch_shape = jnp.broadcast_shapes(H_batch, *Ls_batch, psi0_batch)

    # shape checks
    assert result.states.shape == (*batch_shape, ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*batch_shape, ntrajs, nEs, ntsave)
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
        # trajectories within a batch element must also be distinct
        if ntrajs > 1:
            assert not jnp.array_equal(flat_keys[0, 0], flat_keys[0, 1]), (
                'Trajectory keys within a batch element are identical'
            )


# ---------------------------------------------------------------------------
# Trajectory-independence test: realistic simulation, verify states diverge
# ---------------------------------------------------------------------------
# Two identical Hamiltonians (batch=2) with ntrajs=2 → 4 total trajectories.
# Each (batch, traj) pair gets a unique PRNG key, so all 4 final states must
# differ despite identical physics.  Both cartesian and flat batching are
# tested — with a single batch dim they follow different code paths but yield
# the same output shape.
@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_keys_trajectory_independence(cartesian_batching):
    n = 8
    a = dq.destroy(n)

    # two identical Hamiltonians
    H = (a.dag() @ a).broadcast_to(2, n, n)
    jump_ops = [a]
    psi0 = dq.fock(n, 5)  # high photon number → many jumps guaranteed

    ntrajs = 2
    ntsave = 11
    keys = jax.random.split(jax.random.key(42), ntrajs)
    tsave = jnp.linspace(0.0, 2.0, ntsave)
    options = dq.Options(cartesian_batching=cartesian_batching)
    method = dq.method.Event(dtmax=5e-2)

    result = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, options=options, method=method
    )

    # shape: (2_batch, 2_trajs, ...)
    assert result.states.shape == (2, ntrajs, ntsave, n, 1)
    assert result.keys.shape == (2, ntrajs)

    # all 4 final states must be pairwise different
    final = result.final_state.to_jax().reshape(4, n, 1)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.allclose(final[i], final[j], atol=1e-6), (
                f'Trajectories {i} and {j} have identical final states — '
                'key independence is broken'
            )
