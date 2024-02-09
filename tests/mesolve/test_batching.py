import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_batching(cartesian_batching):
    n = 8
    nt = 11
    nH = 3
    nLs = 4 if cartesian_batching else nH
    npsi0 = 5 if cartesian_batching else nH
    nEs = 6

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    H = dq.rand.herm(k1, (nH, n, n))
    Ls = dq.rand.herm(k2, (nLs, 2, n, n))
    exp_ops = dq.rand.complex(k3, (nEs, n, n))
    psi0 = dq.rand.ket(k4, (npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, nt)

    # no batching
    result = dq.mesolve(H[0], Ls[0], psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (nt, n, n)
    assert result.Esave.shape == (nEs, nt)

    # H batched
    result = dq.mesolve(H, Ls[0], psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (nH, nt, n, n)
    assert result.Esave.shape == (nH, nEs, nt)

    # Ls batched
    result = dq.mesolve(H[0], Ls, psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (nLs, nt, n, n)
    assert result.Esave.shape == (nLs, nEs, nt)

    # psi0 batched
    result = dq.mesolve(H[0], Ls[0], psi0, tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (npsi0, nt, n, n)
    assert result.Esave.shape == (npsi0, nEs, nt)

    # H and Ls batched
    result = dq.mesolve(H, Ls, psi0[0], tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.ysave.shape == (nH, nLs, nt, n, n)
        assert result.Esave.shape == (nH, nLs, nEs, nt)
    else:
        assert result.ysave.shape == (nH, nt, n, n)
        assert result.Esave.shape == (nH, nEs, nt)

    # H and psi0 batched
    result = dq.mesolve(H, Ls[0], psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.ysave.shape == (nH, npsi0, nt, n, n)
        assert result.Esave.shape == (nH, npsi0, nEs, nt)
    else:
        assert result.ysave.shape == (nH, nt, n, n)
        assert result.Esave.shape == (nH, nEs, nt)

    # H, Ls and psi0 batched
    result = dq.mesolve(H, Ls, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.ysave.shape == (nH, nLs, npsi0, nt, n, n)
        assert result.Esave.shape == (nH, nLs, npsi0, nEs, nt)
    else:
        assert result.ysave.shape == (nH, nt, n, n)
        assert result.Esave.shape == (nH, nEs, nt)
