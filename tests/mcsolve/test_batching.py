import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


def rand_mcsolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 4
    kH, *kLs, kpsi0, kEs, kmc = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.rand_herm(kH, (*nH, n, n))
    Ls = [dq.rand_herm(kL, (*nL, n, n)) for kL, nL in zip(kLs, nLs)]
    psi0 = dq.rand_ket(kpsi0, (*npsi0, n, 1))
    Es = dq.rand_complex(kEs, (nEs, n, n))
    return H, Ls, psi0, Es, kmc


@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
@pytest.mark.parametrize('nL1', [(), (7, 8)])
@pytest.mark.parametrize('nL2', [(), (9,)])
def test_cartesian_batching(nH, npsi0, nL1, nL2):
    n = 2
    nLs = [nL1, nL2]
    nEs = 10
    ntsave = 11
    ntraj = 6

    # run mesolve
    H, Ls, psi0, Es, kmc = rand_mcsolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(ntraj=ntraj, save_states=True, one_jump_only=True)
    result = dq.mcsolve(H, Ls, psi0, tsave, key=kmc, exp_ops=Es, options=options)

    # check result shape
    assert result.jump_states.shape == (*nH, *nL1, *nL2, *npsi0, ntraj, ntsave, n, 1)
    assert result.no_jump_states.shape == (*nH, *nL1, *nL2, *npsi0, ntsave, n, 1)
    assert result.expects.shape == (*nH, *nL1, *nL2, *npsi0, nEs, ntsave)


# H has fixed shape (3, 4, n, n) for the next test case, we test a broad ensemble of
# compatible broadcastable shape
@pytest.mark.parametrize('nL1', [(), (5, 1, 4)])
@pytest.mark.parametrize('npsi0', [(), (1,), (4,), (3, 1), (3, 4), (5, 1, 4)])
def test_flat_batching(nL1, npsi0):
    n = 2
    nH = (3, 4)
    nLs = [nL1, ()]
    nEs = 6
    ntsave = 11
    ntraj = 8

    # run mesolve
    H, Ls, psi0, Es, kmc = rand_mcsolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(cartesian_batching=False, ntraj=ntraj)
    result = dq.mcsolve(H, Ls, psi0, tsave, key=kmc, exp_ops=Es, options=options)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1, npsi0)
    assert result.jump_states.shape == (*broadcast_shape, ntraj, ntsave, n, 1)
    assert result.no_jump_states.shape == (*broadcast_shape, ntsave, n, 1)
    assert result.expects.shape == (*broadcast_shape, nEs, ntsave)
