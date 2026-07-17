import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.qarrays.layout import Layout

from ..order import TEST_INSTANT

# batched qarray of shape (5, 3, 4, 4)
_SHAPE = (5, 3, 4, 4)

# keys that only touch the batch dimensions: indexing returns a QArray
BATCH_KEYS = [
    0,
    -1,
    slice(None),
    slice(1, 3),
    (0, 1),
    (0, slice(None)),
    (slice(None), 1),
    ...,
    (0, ...),
    (0, ..., slice(None)),
    (..., slice(None), slice(None)),
    (0, 1, slice(None), slice(None)),
    None,
    (None, 0),
    (0, None, 1),
    (slice(None), slice(None), None),
    jnp.array(1),
    jnp.array([0, 2]),
    (slice(None), jnp.array([0, 2])),
    np.array([True, False, True, False, True]),
    np.ones((5, 3), dtype=bool),
]

# keys that modify the last two dimensions: indexing returns a raw array
MATRIX_KEYS = [
    (0, 1, 2),
    (0, 1, 2, 3),
    (..., 0),
    (..., 0, 0),
    (..., 0, slice(None)),
    (..., slice(0, 2), slice(None)),
    (..., slice(None, None, -1), slice(None)),
    (..., jnp.array([0, 2]), slice(None)),
    (0, 1, slice(None), 2),
    (..., None),
    (slice(None), slice(None), slice(None), None),
    (..., None, slice(None)),
    np.ones((5, 3, 4), dtype=bool),
]


def _qarray(layout: Layout) -> QArray:
    data = jr.normal(jr.key(42), _SHAPE)
    if layout == dq.dia:
        # batch of diagonal matrices, so the dia layout is preserved under stacking
        data = jnp.eye(4) * data[..., :1]
    return dq.asqarray(data, layout=layout)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
@pytest.mark.parametrize('key', BATCH_KEYS)
def test_getitem_batch_dims_returns_qarray(layout, key):
    x = _qarray(layout)
    result = x[key]
    assert isinstance(result, QArray)
    assert result.dims == x.dims
    assert result.shape[-2:] == x.shape[-2:]
    assert jnp.array_equal(result.to_jax(), x.to_jax()[key])


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
@pytest.mark.parametrize('key', MATRIX_KEYS)
def test_getitem_matrix_dims_returns_array(layout, key):
    x = _qarray(layout)
    result = x[key]
    assert isinstance(result, Array)
    assert jnp.array_equal(result, x.to_jax()[key])


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_getitem_unbatched(layout):
    x = _qarray(layout)[0, 0]
    assert isinstance(x[:], QArray)
    assert isinstance(x[...], QArray)
    assert isinstance(x[None], QArray)
    assert isinstance(x[0], Array)
    assert isinstance(x[0, 1], Array)
    assert isinstance(x[:, 0], Array)
