import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


def rand_mesolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 3
    kH, *kLs, kpsi0, kEs = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.rand_herm(kH, (*nH, n, n))
    Ls = [dq.rand_herm(kL, (*nL, n, n)) for kL, nL in zip(kLs, nLs)]
    psi0 = dq.rand_ket(kpsi0, (*npsi0, n, 1))
    Es = dq.rand_complex(kEs, (nEs, n, n))
    return H, Ls, psi0, Es


@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
@pytest.mark.parametrize('nL1', [(), (7, 8)])
@pytest.mark.parametrize('nL2', [(), (9,)])
def test_cartesian_batching(nH, npsi0, nL1, nL2):
    n = 2
    nLs = [nL1, nL2]
    nEs = 10
    ntsave = 11

    # run mesolve
    H, Ls, psi0, Es = rand_mesolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=Es)

    # check result shape
    assert result.states.shape == (*nH, *nL1, *nL2, *npsi0, ntsave, n, n)
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

    # run mesolve
    H, Ls, psi0, Es = rand_mesolve_args(n, nH, nLs, npsi0, nEs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(cartesian_batching=False)
    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=Es, options=options)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1, npsi0)
    assert result.states.shape == (*broadcast_shape, ntsave, n, n)
    assert result.expects.shape == (*broadcast_shape, nEs, ntsave)
