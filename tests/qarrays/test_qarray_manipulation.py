from dataclasses import replace

import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.fixture
def batched_data():
    return jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 4, 4)


@pytest.fixture
def batched_qarray(batched_data):
    return dq.asqarray(batched_data, dims=(2, 2))


@pytest.mark.run(order=TEST_INSTANT)
def test_swapaxes_dense_matches_jax(batched_data, batched_qarray):
    result = dq.swapaxes(batched_qarray, 0, 1)

    assert result.shape == (3, 2, 4, 4)
    assert result.dims == (2, 2)
    assert result.dtype == batched_qarray.dtype
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.swapaxes(batched_data, 0, 1))

    mt = batched_qarray.swapaxes(-1, -2)
    assert jnp.array_equal(mt.to_jax(), jnp.swapaxes(batched_data, -1, -2))


@pytest.mark.run(order=TEST_INSTANT)
def test_moveaxis_dense_matches_jax(batched_data, batched_qarray):
    result = batched_qarray.moveaxis(0, 1)

    assert result.shape == (3, 2, 4, 4)
    assert result.dims == (2, 2)
    assert result.dtype == batched_qarray.dtype
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.moveaxis(batched_data, 0, 1))

    mt = dq.moveaxis(batched_qarray, -1, -2)
    assert jnp.array_equal(mt.to_jax(), jnp.moveaxis(batched_data, -1, -2))


@pytest.mark.run(order=TEST_INSTANT)
def test_moveaxis_dense_multi_axis_matches_jax():
    data = jnp.arange(2 * 3 * 5 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 5, 4, 4)
    qarray = dq.asqarray(data, dims=(2, 2))

    result = qarray.moveaxis((0, 1), (2, 0))

    assert result.shape == (3, 5, 2, 4, 4)
    assert result.dims == (2, 2)
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.moveaxis(data, (0, 1), (2, 0)))


@pytest.mark.run(order=TEST_INSTANT)
def test_expand_dims_dense_matches_jax(batched_data, batched_qarray):
    result = dq.expand_dims(batched_qarray, axis=(0, 2))

    assert result.shape == (1, 2, 1, 3, 4, 4)
    assert result.dims == (2, 2)
    assert result.dtype == batched_qarray.dtype
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.expand_dims(batched_data, axis=(0, 2)))


@pytest.mark.run(order=TEST_INSTANT)
def test_where_matches_jax_and_accepts_scalars():
    x_data = jnp.arange(2 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3)
    y_data = -x_data
    condition = jnp.asarray([True, False])[:, None, None]
    x = dq.asqarray(x_data)
    y = dq.asqarray(y_data)

    result = dq.where(condition, x, y)

    assert result.shape == (2, 3, 3)
    assert result.dims == (3,)
    assert result.dtype == x.dtype
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.where(condition, x_data, y_data))

    scalar_result = x.where(condition, 0.0)
    assert jnp.array_equal(scalar_result.to_jax(), jnp.where(condition, x_data, 0.0))


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_dense_matches_jax():
    x_data = jnp.arange(2 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3)
    y_data = -x_data
    x = dq.asqarray(x_data)
    y = dq.asqarray(y_data)

    result = dq.concatenate([x, y], axis=0)

    assert result.shape == (4, 3, 3)
    assert result.dims == (3,)
    assert result.dtype == x.dtype
    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.concatenate([x_data, y_data], axis=0))


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('operation', ['swapaxes', 'moveaxis', 'expand_dims'])
def test_vectorized_axis_manipulation_preserves_vectorized(operation):
    data = jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 4, 4)
    qarray = dq.vectorize(dq.asqarray(data, dims=(2, 2)))

    if operation == 'swapaxes':
        result = qarray.swapaxes(0, 1)
        expected = jnp.swapaxes(qarray.to_jax(), 0, 1)
    elif operation == 'moveaxis':
        result = qarray.moveaxis(0, 1)
        expected = jnp.moveaxis(qarray.to_jax(), 0, 1)
    else:
        result = qarray.expand_dims(0)
        expected = jnp.expand_dims(qarray.to_jax(), 0)

    assert result.vectorized
    assert result.shape == expected.shape
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_vectorized_concatenate_preserves_vectorized():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    y_data = -x_data
    x = dq.vectorize(dq.asqarray(x_data, dims=(2, 2)))
    y = dq.vectorize(dq.asqarray(y_data, dims=(2, 2)))

    result = dq.concatenate([x, y], axis=0)

    assert result.vectorized
    assert jnp.array_equal(result.to_jax(), jnp.concatenate([x.to_jax(), y.to_jax()]))


@pytest.mark.run(order=TEST_INSTANT)
def test_vectorized_where_preserves_vectorized():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    y_data = -x_data
    condition = jnp.asarray([True, False])[:, None, None]
    x = dq.vectorize(dq.asqarray(x_data, dims=(2, 2)))
    y = dq.vectorize(dq.asqarray(y_data, dims=(2, 2)))

    result = dq.where(condition, x, y)

    assert result.vectorized
    assert jnp.array_equal(
        result.to_jax(), jnp.where(condition, x.to_jax(), y.to_jax())
    )


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('operation', ['swapaxes', 'moveaxis', 'expand_dims'])
def test_sparse_batch_axis_manipulation_preserves_sparse_layout(operation):
    data = jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 4, 4)
    qarray = dq.asqarray(data, dims=(4,), layout=dq.dia)

    if operation == 'swapaxes':
        result = dq.swapaxes(qarray, 0, 1)
        expected = jnp.swapaxes(data, 0, 1)
    elif operation == 'moveaxis':
        result = dq.moveaxis(qarray, 0, 1)
        expected = jnp.moveaxis(data, 0, 1)
    else:
        result = dq.expand_dims(qarray, 0)
        expected = jnp.expand_dims(data, 0)

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_swapaxes_final_axes_preserves_sparse_layout():
    data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    qarray = dq.asqarray(data, layout=dq.dia)

    result = dq.swapaxes(qarray, -1, -2)

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), jnp.swapaxes(data, -1, -2))


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_moveaxis_multi_axis_matches_jax():
    data = jnp.arange(2 * 3 * 5 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 5, 4, 4)
    qarray = dq.asqarray(data, dims=(4,), layout=dq.dia)

    result = qarray.moveaxis((0, 1), (2, 0))

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), jnp.moveaxis(data, (0, 1), (2, 0)))


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_expand_dims_tuple_axis_matches_jax():
    data = jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 4, 4)
    qarray = dq.asqarray(data, dims=(4,), layout=dq.dia)

    result = qarray.expand_dims(axis=(0, 2))

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), jnp.expand_dims(data, axis=(0, 2)))


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_concatenate_batch_axis_preserves_sparse_layout():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    y_data = -x_data
    x = dq.asqarray(x_data, layout=dq.dia)
    y = dq.asqarray(y_data, layout=dq.dia)

    result = dq.concatenate([x, y], axis=0)

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), jnp.concatenate([x_data, y_data], axis=0))


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_concatenate_merges_different_offsets():
    x = dq.sparsedia_from_dict({0: jnp.ones((2, 4))})
    y = dq.sparsedia_from_dict({1: jnp.ones((3, 3))})

    result = dq.concatenate([x, y], axis=0)

    assert result.layout == dq.dia
    assert result.data.offsets == (0, 1)
    assert jnp.array_equal(
        result.to_jax(), jnp.concatenate([x.to_jax(), y.to_jax()], axis=0)
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_sparse_moveaxis_final_axes_preserves_sparse_layout():
    data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    qarray = dq.asqarray(data, layout=dq.dia)

    result = dq.moveaxis(qarray, -1, -2)

    assert result.layout == dq.dia
    assert jnp.array_equal(result.to_jax(), jnp.moveaxis(data, -1, -2))


@pytest.mark.run(order=TEST_INSTANT)
def test_moveaxis_can_swap_final_axes_while_moving_batch_axes():
    qarray = dq.eye(4).broadcast_to(1, 1, 4, 4)

    result = qarray.moveaxis((0, -1, -2), (1, -2, -1))

    assert result.layout == dq.dia
    assert jnp.array_equal(
        result.to_jax(), jnp.moveaxis(qarray.to_jax(), (0, -1, -2), (1, -2, -1))
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_where_sparse_converts_to_dense_with_warning():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    y_data = -x_data
    condition = jnp.asarray([True, False])[:, None, None]
    x = dq.asqarray(x_data, layout=dq.dia)
    y = dq.asqarray(y_data, layout=dq.dia)

    with pytest.warns(UserWarning, match='converted to dense layout'):
        result = dq.where(condition, x, y)

    assert result.layout == dq.dense
    assert jnp.array_equal(result.to_jax(), jnp.where(condition, x_data, y_data))


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('operation', 'args'),
    [
        ('swapaxes', (0, -1)),
        ('moveaxis', (0, -1)),
        ('expand_dims', (-1,)),
        ('concatenate', (-1,)),
    ],
)
def test_quantum_axis_manipulation_raises(batched_qarray, operation, args):
    match = 'final two dimensions'

    with pytest.raises(ValueError, match=match):
        _apply_quantum_axis_operation(batched_qarray, operation, args)


@pytest.mark.run(order=TEST_INSTANT)
def test_where_rejects_incompatible_quantum_shapes():
    operator = dq.eye(3)
    ket = dq.fock(3, 0)

    with pytest.raises(ValueError, match='final two dimensions'):
        dq.where(True, operator, ket)


@pytest.mark.run(order=TEST_INSTANT)
def test_where_requires_qarray_operand():
    with pytest.raises(TypeError, match=r'jax\.numpy\.where'):
        dq.where(True, 1.0, 2.0)


@pytest.mark.run(order=TEST_INSTANT)
def test_where_rejects_mismatched_vectorized_metadata():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    x = dq.vectorize(dq.asqarray(x_data, dims=(2, 2)))
    y = replace(x, vectorized=False)

    with pytest.raises(ValueError, match='incompatible `vectorized`'):
        dq.where(True, x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_rejects_incompatible_dims():
    x = dq.eye(2)
    y = dq.eye(3)

    with pytest.raises(ValueError, match='identical `dims`'):
        dq.concatenate([dq.stack([x]), dq.stack([y])], axis=0)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_rejects_non_qarray_input():
    with pytest.raises(ValueError, match='type `QArray`'):
        dq.concatenate([jnp.ones((2, 2)), jnp.ones((2, 2))], axis=0)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_rejects_incompatible_batch_ndim():
    x = dq.stack([dq.eye(2)])
    y = dq.stack([dq.stack([dq.eye(2)])])

    with pytest.raises(ValueError, match='batching dimensions'):
        dq.concatenate([x, y], axis=0)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_rejects_mismatched_vectorized_metadata():
    x_data = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    x = dq.vectorize(dq.asqarray(x_data, dims=(2, 2)))
    y = replace(x, vectorized=False)

    with pytest.raises(ValueError, match='identical `vectorized`'):
        dq.concatenate([x, y], axis=0)


def _apply_quantum_axis_operation(qarray, operation, args):
    if operation == 'swapaxes':
        return dq.swapaxes(qarray, *args)
    elif operation == 'moveaxis':
        return dq.moveaxis(qarray, *args)
    elif operation == 'expand_dims':
        return dq.expand_dims(qarray, *args)
    else:
        return dq.concatenate([qarray, qarray], axis=args[0])
