import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq

from ..order import TEST_LONG
from .mepropagator_utils import rand_mepropagator_args


@pytest.mark.run(order=TEST_LONG)
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
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nL1', [(), (5, 1, 4)])
def test_flat_batching(nL1):
    n = 2
    nH = (3, 4)
    nLs = [nL1, ()]
    ntsave = 11

    # run mepropagator
    H, Ls = rand_mepropagator_args(n, nH, nLs)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.mepropagator(H, Ls, tsave, cartesian_batching=False)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1)
    assert result.propagators.shape == (*broadcast_shape, ntsave, n**2, n**2)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(('nH', 'nL2'), [((3,), ()), ((), (3,)), ((2, 1), (3,))])
def test_flat_batching_values(nH, nL2):
    # flat batching, with unbatched or partially batched inputs, gives the same
    # propagators as computing each batch element on its own
    H, (L1, L2) = rand_mepropagator_args(2, nH, [(), nL2])
    tsave = jnp.linspace(0, 0.01, 5)
    result = dq.mepropagator(H, [L1, L2], tsave, cartesian_batching=False)
    for index in np.ndindex(jnp.broadcast_shapes(nH, nL2)):
        expected = dq.mepropagator(
            _element(H, nH, index), [L1, _element(L2, nL2, index)], tsave
        )
        assert jnp.allclose(
            result.propagators[index].to_jax(), expected.propagators.to_jax()
        )


def _element(x, shape, index):
    # the inputs of batch element `index` of a flat batch, with broadcasting
    if len(shape) == 0:
        return x
    return x[
        tuple(
            i if d > 1 else 0 for i, d in zip(index[-len(shape) :], shape, strict=True)
        )
    ]
