import jax
import pytest
from jax import numpy as jnp

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_batching(cartesian_batching):
    n = 8
    nt = 11
    nH = 3
    npsi0 = 4 if cartesian_batching else nH
    nEs = 5

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    key = jax.random.PRNGKey(42)
    H = dq.rand.herm(key, (nH, n, n))
    exp_ops = dq.rand.complex(key, (nEs, n, n))
    psi0 = dq.rand.ket(key, (npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, nt)

    # no batching
    result = dq.sesolve(H[0], psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (nt, n, 1)
    assert result.Esave.shape == (nEs, nt)

    # H batched
    result = dq.sesolve(H, psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (nH, nt, n, 1)
    assert result.Esave.shape == (nH, nEs, nt)

    # psi0 batched
    result = dq.sesolve(H[0], psi0, tsave, exp_ops=exp_ops, options=options)
    assert result.ysave.shape == (npsi0, nt, n, 1)
    assert result.Esave.shape == (npsi0, nEs, nt)

    # H and psi0 batched
    result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.ysave.shape == (nH, npsi0, nt, n, 1)
        assert result.Esave.shape == (nH, npsi0, nEs, nt)
    else:
        assert result.ysave.shape == (nH, nt, n, 1)
        assert result.Esave.shape == (nH, nEs, nt)
