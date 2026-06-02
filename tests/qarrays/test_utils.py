import jax.numpy as jnp
import jax.random as jr
import pytest
import qutip as qt

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_stack_simple(rtol=1e-05, atol=1e-08):
    n = 10
    key = jr.key(42)

    jax_m = jr.uniform(key, (n, n))
    dense_m = dq.asqarray(jax_m, layout=dq.dense)
    sparse_m = dq.asqarray(jax_m, layout=dq.dia)

    assert jnp.allclose(
        dq.stack([dense_m, 2 * dense_m]).to_jax(),
        jnp.stack([jax_m, 2 * jax_m]),
        rtol=rtol,
        atol=atol,
    )

    assert jnp.allclose(
        dq.stack([sparse_m, 2 * sparse_m]).to_jax(),
        jnp.stack([jax_m, 2 * jax_m]),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_stack_double(rtol=1e-05, atol=1e-08):
    n = 10
    key = jr.key(42)
    k1, k2 = jr.split(key)

    jax_m1 = jr.uniform(k1, (n, n))
    dense_m1 = dq.asqarray(jax_m1, layout=dq.dense)
    sparse_m1 = dq.asqarray(jax_m1, layout=dq.dia)
    dense_m1 = dq.stack([dense_m1, 2 * dense_m1, 3 * dense_m1])
    sparse_m1 = dq.stack([sparse_m1, 2 * sparse_m1, 3 * sparse_m1])
    jax_m1 = jnp.stack([jax_m1, 2 * jax_m1, 3 * jax_m1])

    jax_m2 = jr.uniform(k2, (n, n))
    dense_m2 = dq.asqarray(jax_m2, layout=dq.dense)
    sparse_m2 = dq.asqarray(jax_m2, layout=dq.dia)
    dense_m2 = dq.stack([dense_m2, 2 * dense_m2, 3 * dense_m2])
    sparse_m2 = dq.stack([sparse_m2, 2 * sparse_m2, 3 * sparse_m2])
    jax_m2 = jnp.stack([jax_m2, 2 * jax_m2, 3 * jax_m2])

    assert jnp.allclose(
        dq.stack([dense_m1, dense_m2]).to_jax(),
        jnp.stack([jax_m1, jax_m2]),
        rtol=rtol,
        atol=atol,
    )

    assert jnp.allclose(
        dq.stack([sparse_m1, sparse_m2]).to_jax(),
        jnp.stack([jax_m1, jax_m2]),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_qarray_manipulation_preserves_batch_qarray(layout):
    data = jnp.arange(24).reshape(2, 3, 2, 2)
    qarray = dq.asqarray(data, dims=(2,), layout=layout)

    swapped = dq.swapaxes(qarray, 0, 1)
    assert isinstance(swapped, dq.QArray)
    assert swapped.dims == qarray.dims
    assert jnp.array_equal(swapped.to_jax(), jnp.swapaxes(data, 0, 1))

    moved = dq.moveaxis(qarray, 0, 1)
    assert isinstance(moved, dq.QArray)
    assert moved.dims == qarray.dims
    assert jnp.array_equal(moved.to_jax(), jnp.moveaxis(data, 0, 1))

    expanded = dq.expand_dims(qarray, 1)
    assert isinstance(expanded, dq.QArray)
    assert expanded.dims == qarray.dims
    assert jnp.array_equal(expanded.to_jax(), jnp.expand_dims(data, 1))

    assert jnp.array_equal(qarray.swapaxes(0, 1).to_jax(), jnp.swapaxes(data, 0, 1))
    assert jnp.array_equal(qarray.moveaxis(0, 1).to_jax(), jnp.moveaxis(data, 0, 1))
    assert jnp.array_equal(qarray.expand_dims(1).to_jax(), jnp.expand_dims(data, 1))


@pytest.mark.run(order=TEST_INSTANT)
def test_qarray_manipulation_returns_array_when_quantum_shape_changes():
    qarray = dq.asqarray(jnp.arange(24).reshape(2, 3, 2, 2), dims=(2,))

    moved = dq.moveaxis(qarray, -1, 0)

    assert not isinstance(moved, dq.QArray)
    assert jnp.array_equal(moved, jnp.moveaxis(qarray.to_jax(), -1, 0))


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_qarray_where_and_concatenate(layout):
    x = dq.asqarray(jnp.arange(8).reshape(2, 2, 2), dims=(2,), layout=layout)
    y = dq.asqarray(jnp.arange(8, 16).reshape(2, 2, 2), dims=(2,), layout=layout)

    condition = jnp.array([True, False])[:, None, None]
    selected = dq.where(condition, x, y)

    assert isinstance(selected, dq.QArray)
    assert selected.dims == x.dims
    assert jnp.array_equal(
        selected.to_jax(), jnp.where(condition, x.to_jax(), y.to_jax())
    )

    selected_with_scalar = dq.where(condition, x, 0)
    assert isinstance(selected_with_scalar, dq.QArray)
    assert jnp.array_equal(
        selected_with_scalar.to_jax(), jnp.where(condition, x.to_jax(), 0)
    )

    concatenated = dq.concatenate([x, y], axis=0)
    assert isinstance(concatenated, dq.QArray)
    assert concatenated.dims == x.dims
    assert jnp.array_equal(
        concatenated.to_jax(), jnp.concatenate([x.to_jax(), y.to_jax()], axis=0)
    )


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_conversions(layout):
    sx, sy = dq.sigmax(), dq.sigmay()
    assert jnp.allclose(sx.to_jax(), dq.asqarray(sx, layout=layout).to_jax())
    assert jnp.allclose(
        dq.stack([sx, sy]).to_jax(), dq.asqarray([sx, sy], layout=layout).to_jax()
    )
    assert jnp.allclose(
        jnp.asarray([sx.to_jax(), sy.to_jax()]),
        dq.asqarray([sx, sy], layout=layout).to_jax(),
    )
    assert jnp.allclose(
        sx.to_jax(), dq.asqarray(sx.to_jax().tolist(), layout=layout).to_jax()
    )
    assert jnp.allclose(
        jnp.asarray([[sx.to_jax(), sy.to_jax()]]),
        dq.asqarray([(sx.to_jax().tolist(), sy)], layout=layout).to_jax(),
    )
    assert jnp.allclose(dq.asqarray(sx.to_qutip(), layout=layout).to_jax(), sx.to_jax())
    assert jnp.allclose(
        jnp.asarray(
            [x.full() for x in dq.asqarray([sx, sy], layout=layout).to_qutip()]
        ),
        jnp.asarray([sx.to_qutip().full(), sy.to_qutip().full()]),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_qutip_tensor_compatibility():
    """Test compatibility with qutip v5.2.0 auto_tidyup_dims.

    In qutip v5.2.0, tensor products with trivial dimensions are automatically
    simplified, e.g., [[3, 2], [1, 1]] -> [[3, 2], [1]]. This test ensures
    that asqarray handles such cases correctly.
    """
    # Test single basis state (should work in all versions)
    basis_state = qt.basis(3, 1)
    result = dq.asqarray(basis_state)
    assert result.shape == (3, 1)
    assert result.dims == (3,)

    # Test tensor product (the problematic case)
    tensor_state = qt.tensor(qt.basis(3, 0), qt.basis(2, 1))
    result = dq.asqarray(tensor_state)
    assert result.shape == (6, 1)
    assert result.dims == (3, 2)

    # Test more complex tensor product
    complex_tensor = qt.tensor(qt.basis(2, 0), qt.basis(3, 1), qt.basis(2, 0))
    result = dq.asqarray(complex_tensor)
    assert result.shape == (12, 1)
    assert result.dims == (2, 3, 2)

    # Test that the tensor product has correct values
    tensor_result = dq.asqarray(qt.tensor(qt.basis(3, 0), qt.basis(2, 1)))
    expected_tensor = jnp.zeros((6, 1), dtype=complex)
    expected_tensor = expected_tensor.at[1].set(1.0)  # |01⟩ state
    assert jnp.allclose(tensor_result.to_jax(), expected_tensor)
