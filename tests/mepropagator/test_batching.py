import jax.numpy as jnp
import pytest

import dynamiqs as dq

from .mepropagator_utils import rand_mepropagator_args


@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('nL1', [(), (7, 8)])
@pytest.mark.parametrize('nL2', [(), (9,)])
def test_cartesian_batching(nH, nL1, nL2):
    n = 2
    nLs = [nL1, nL2]
    ntsave = 11

    # run mepropagator
    H, Ls = rand_mepropagator_args(n, nH, nLs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.mepropagator(H, Ls, tsave)

    # check result shape
    assert result.propagators.shape == (*nH, *nL1, *nL2, ntsave, n**2, n**2)


# H has fixed shape (3, 4, n, n) for the next test case, we test flat batching
# of jump operators
@pytest.mark.parametrize('nL1', [(), (5, 1, 4)])
def test_flat_batching(nL1):
    n = 2
    nH = (3, 4)
    nLs = [nL1, ()]
    ntsave = 11

    # run mepropagator
    H, Ls = rand_mepropagator_args(n, nH, nLs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    options = dq.Options(cartesian_batching=False)
    result = dq.mepropagator(H, Ls, tsave, options=options)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1)
    assert result.propagators.shape == (*broadcast_shape, ntsave, n**2, n**2)
