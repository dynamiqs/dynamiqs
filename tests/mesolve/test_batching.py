import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,), (5, 6)])
@pytest.mark.parametrize('nL1', [(), (7,), (7, 8)])
@pytest.mark.parametrize('nL2', [(), (9,)])
def test_batching(cartesian_batching, nH, npsi0, nL1, nL2):
    n = 8
    ntsave = 11
    nL1 = nL1 if cartesian_batching else nH
    nL2 = nL2 if cartesian_batching else nH
    npsi0 = npsi0 if cartesian_batching else nH
    nE = 7

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(42), 5)
    H = dq.rand_herm(k1, (*nH, n, n))
    Ls = [dq.rand_herm(k2, (*nL1, n, n)), dq.rand_herm(k3, (*nL2, n, n))]
    exp_ops = dq.rand_complex(k4, (nE, n, n))
    psi0 = dq.rand_ket(k5, (*npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, ntsave)

    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (*nH, *nL1, *nL2, *npsi0, ntsave, n, n)
        assert result.expects.shape == (*nH, *nL1, *nL2, *npsi0, nE, ntsave)
    else:
        assert result.states.shape == (*nH, ntsave, n, n)
        assert result.expects.shape == (*nH, nE, ntsave)


@pytest.mark.parametrize(('nH'), [(), (3,), (4, 3)])
@pytest.mark.parametrize(('npsi0'), [(), (3,), (4, 3)])
@pytest.mark.parametrize(('nL'), [(), (3,), (4, 3)])
def test_non_carthesian_batching_broacasting(nH, npsi0, nL):
    n = 8
    nE = 7
    ntsave = 11

    options = dq.options.Options(cartesian_batching=False)

    # create random objects
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    H = dq.rand_herm(k1, (*nH, n, n))
    exp_ops = dq.rand_complex(k2, (nE, n, n))
    psi0 = dq.rand_ket(k3, (*npsi0, n, 1))
    Ls = [dq.rand_herm(k2, (*nL, n, n))]
    tsave = jnp.linspace(0, 0.01, ntsave)

    broadcast_shape = jnp.broadcast_shapes(
        psi0.shape[:-2], H.shape[:-2], *[L.shape[:-2] for L in Ls]
    )

    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (*broadcast_shape, ntsave, n, n)
    assert result.expects.shape == (*broadcast_shape, nE, ntsave)
