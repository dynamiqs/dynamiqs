import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import dynamiqs as dq
from dynamiqs import QArray
from dynamiqs.qarrays.composite_qarray import CompositeQArray, CompositeTerm

from ..order import TEST_INSTANT

# batched qarray of shape (5, 3, 4, 4)
_SHAPE = (5, 3, 4, 4)

_LAYOUTS = {'dense': dq.dense, 'dia': dq.dia}

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


def _qarray(kind: str) -> QArray:
    if kind == 'composite':
        key0, key1 = jr.split(jr.key(44))
        A = dq.asqarray(jr.normal(key0, (5, 3, 2, 2)), dims=(2,))
        B = dq.asqarray(jr.normal(key1, (2, 2)), dims=(2,))
        return CompositeQArray((2, 2), (CompositeTerm(operators=(A, B)),))

    data = jr.normal(jr.key(42), _SHAPE)
    if kind == 'dia':
        # batch of diagonal matrices, so the dia layout is preserved under stacking
        data = jnp.eye(4) * data[..., :1]
    return dq.asqarray(data, layout=_LAYOUTS[kind])


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('kind', ['dense', 'dia', 'composite'])
@pytest.mark.parametrize('key', BATCH_KEYS)
def test_getitem_batch_dims_returns_qarray(kind, key):
    x = _qarray(kind)
    result = x[key]
    assert isinstance(result, QArray)
    assert result.dims == x.dims
    assert result.shape[-2:] == x.shape[-2:]
    assert jnp.array_equal(result.to_jax(), x.to_jax()[key])


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('kind', ['dense', 'dia', 'composite'])
@pytest.mark.parametrize('key', MATRIX_KEYS)
def test_getitem_matrix_dims_returns_array(kind, key):
    x = _qarray(kind)
    result = x[key]
    assert isinstance(result, Array)
    assert jnp.array_equal(result, x.to_jax()[key])


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('kind', ['dense', 'dia', 'composite'])
def test_getitem_unbatched(kind):
    x = _qarray(kind)[0, 0]
    assert isinstance(x[:], QArray)
    assert isinstance(x[...], QArray)
    assert isinstance(x[None], QArray)
    assert isinstance(x[0], Array)
    assert isinstance(x[0, 1], Array)
    assert isinstance(x[:, 0], Array)
