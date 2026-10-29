import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


def rand_steadystate_args(n, nH, nLs):
    kH, *kLs = jax.random.split(jax.random.PRNGKey(42), len(nLs) + 1)
    H = dq.random.herm(kH, (*nH, n, n))
    Ls = [dq.random.operator(kL, n, batch=nL) for kL, nL in zip(kLs, nLs, strict=True)]
    return H, Ls


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('nL1', [(), (5, 6)])
@pytest.mark.parametrize('nL2', [(), (7,)])
def test_cartesian_batching(nH, nL1, nL2):
    n, nE = 2, 3
    H, Ls = rand_steadystate_args(n, nH, [nL1, nL2])
    exp_ops = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]
    result = dq.steadystate(H, Ls, exp_ops=exp_ops)

    batch = (*nH, *nL1, *nL2)
    assert result.state.shape == (*batch, n, n)
    assert result.expects.shape == (*batch, nE)
    assert result.infos.converged.shape == batch


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nL1', [(), (5, 1, 4)])
def test_flat_batching(nL1):
    n, nH = 2, (3, 4)
    H, Ls = rand_steadystate_args(n, nH, [nL1, ()])
    result = dq.steadystate(H, Ls, cartesian_batching=False)

    batch = jnp.broadcast_shapes(nH, nL1)
    assert result.state.shape == (*batch, n, n)
    assert result.infos.converged.shape == batch
