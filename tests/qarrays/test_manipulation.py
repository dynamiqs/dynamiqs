import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.fixture(params=[dq.dense, dq.dia])
def batched_qarray(request):
    data = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 2, 2) + 1j).astype(
        jnp.complex64
    )
    return data, dq.asqarray(data, dims=(2,), layout=request.param)


@pytest.mark.run(order=TEST_INSTANT)
def test_swapaxes_matches_jax_and_preserves_qarray(batched_qarray):
    data, qarray = batched_qarray

    result = dq.swapaxes(qarray, 0, 1)
    expected = jnp.swapaxes(data, 0, 1)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)
    assert jnp.array_equal(qarray.swapaxes(0, 1).to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_swapaxes_quantum_axes_matches_jax(batched_qarray):
    data, qarray = batched_qarray

    result = dq.swapaxes(qarray, -1, -2)
    expected = jnp.swapaxes(data, -1, -2)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_moveaxis_matches_jax_and_preserves_qarray(batched_qarray):
    data, qarray = batched_qarray

    result = dq.moveaxis(qarray, 0, 1)
    expected = jnp.moveaxis(data, 0, 1)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)
    assert jnp.array_equal(qarray.moveaxis(0, 1).to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_moveaxis_returns_array_when_quantum_shape_is_not_preserved(batched_qarray):
    data, qarray = batched_qarray

    result = dq.moveaxis(qarray, -1, 0)
    expected = jnp.moveaxis(data, -1, 0)

    assert not isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert jnp.array_equal(result, expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_expand_dims_matches_jax_and_preserves_qarray(batched_qarray):
    data, qarray = batched_qarray

    result = dq.expand_dims(qarray, 1)
    expected = jnp.expand_dims(data, 1)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)
    assert jnp.array_equal(qarray.expand_dims(1).to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_where_matches_jax_and_preserves_qarray(batched_qarray):
    data, qarray = batched_qarray
    other = dq.asqarray(data + 100, dims=qarray.dims)
    condition = (data.real % 2) == 0

    result = dq.where(condition, qarray, other)
    expected = jnp.where(condition, data, data + 100)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_where_mixed_qarray_and_scalar_matches_jax(batched_qarray):
    data, qarray = batched_qarray
    condition = (data.real % 2) == 0

    result = dq.where(condition, qarray, 0.0)
    expected = jnp.where(condition, data, 0.0)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_matches_jax_and_preserves_qarray(batched_qarray):
    data, qarray = batched_qarray
    other = dq.asqarray(data + 100, dims=qarray.dims)

    result = dq.concatenate([qarray, other], axis=0)
    expected = jnp.concatenate([data, data + 100], axis=0)

    assert isinstance(result, dq.QArray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.dims == qarray.dims
    assert jnp.array_equal(result.to_jax(), expected)


@pytest.mark.run(order=TEST_INSTANT)
def test_concatenate_raw_arrays_matches_jax():
    data = jnp.arange(12).reshape(3, 2, 2)

    result = dq.concatenate([data, data + 100], axis=0)
    expected = jnp.concatenate([data, data + 100], axis=0)

    assert not isinstance(result, dq.QArray)
    assert jnp.array_equal(result, expected)
