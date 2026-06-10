import jax.numpy as jnp
import pytest

from dynamiqs import asqarray, tensor

from ..order import TEST_SHORT


@pytest.mark.run(order=TEST_SHORT)
class TestDenseQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = jnp.arange(16).reshape(4, 4) * (1 + 1j)
        self.qarray = asqarray(self.data, dims=(2, 2))
        self.other = jnp.arange(16).reshape(4, 4) + 16
        self.qother = asqarray(self.other, dims=(2, 2))
        self.scalar = 2 + 2j
        self.bscalar = jnp.ones((3, 2, 1, 1), dtype=jnp.complex64)

    def test_dag(self):
        assert jnp.array_equal(self.qarray.dag().to_jax(), self.data.mT.conj())

    def test_ptrace(self):
        ptrace = self.qarray.ptrace(1)
        expected_ptrace = jnp.array([[10, 12], [18, 20]]) * (1 + 1j)
        assert jnp.array_equal(ptrace.to_jax(), expected_ptrace)
        assert ptrace.dims == (2,)

    def test_add(self):
        assert jnp.array_equal(
            (self.qarray + self.other).to_jax(), self.data + self.other
        )
        assert jnp.array_equal(
            (self.qarray + self.qother).to_jax(), self.data + self.other
        )
        with pytest.raises(NotImplementedError):
            self.qarray + self.scalar
        with pytest.raises(NotImplementedError):
            self.qarray + self.bscalar

    def test_radd(self):
        assert jnp.array_equal(
            (self.other + self.qarray).to_jax(), self.other + self.data
        )
        assert jnp.array_equal(
            (self.qother + self.qarray).to_jax(), self.other + self.data
        )
        with pytest.raises(NotImplementedError):
            self.scalar + self.qarray
        with pytest.raises(NotImplementedError):
            self.bscalar + self.qarray

    def test_scalaradd(self):
        assert jnp.array_equal(
            self.qarray.addscalar(self.scalar).to_jax(), self.data + self.scalar
        )
        assert jnp.array_equal(
            self.qarray.addscalar(self.bscalar).to_jax(), self.data + self.bscalar
        )

    def test_sub(self):
        assert jnp.array_equal(
            (self.qarray - self.other).to_jax(), self.data - self.other
        )
        assert jnp.array_equal(
            (self.qarray - self.qother).to_jax(), self.data - self.other
        )
        with pytest.raises(NotImplementedError):
            self.qarray - self.scalar
        with pytest.raises(NotImplementedError):
            self.qarray - self.bscalar

    def test_mul(self):
        assert jnp.array_equal(
            (self.qarray * self.scalar).to_jax(), self.data * self.scalar
        )
        assert jnp.array_equal(
            (self.qarray * self.bscalar).to_jax(), self.data * self.bscalar
        )

    def test_rmul(self):
        assert jnp.array_equal(
            (self.scalar * self.qarray).to_jax(), self.scalar * self.data
        )
        assert jnp.array_equal(
            (self.bscalar * self.qarray).to_jax(), self.bscalar * self.data
        )

    def test_elmul(self):
        assert jnp.array_equal(
            self.qarray.elmul(self.other).to_jax(), self.data * self.other
        )
        assert jnp.array_equal(
            self.qarray.elmul(self.qother).to_jax(), self.data * self.other
        )

    def test_matmul(self):
        assert jnp.array_equal(
            (self.qarray @ self.other).to_jax(), self.data @ self.other
        )
        assert jnp.array_equal(
            (self.qarray @ self.qother).to_jax(), self.data @ self.other
        )

    def test_rmatmul(self):
        assert jnp.array_equal(
            (self.other @ self.qarray).to_jax(), self.other @ self.data
        )
        assert jnp.array_equal(
            (self.qother @ self.qarray).to_jax(), self.other @ self.data
        )

    def test_and(self):
        t = self.qarray & self.qother

        assert jnp.array_equal(t.to_jax(), tensor(self.data, self.other).to_jax())
        assert t.dims == (2, 2, 2, 2)

        other = jnp.arange(9).reshape(3, 3)
        qother = asqarray(other)
        t = self.qarray & qother

        assert jnp.array_equal(t.to_jax(), tensor(self.data, other).to_jax())
        assert t.dims == (2, 2, 3)

    def test_powm(self):
        assert jnp.array_equal(self.qarray.powm(2).to_jax(), self.data @ self.data)
        assert jnp.array_equal(
            self.qarray.powm(3).to_jax(), self.data @ self.data @ self.data
        )

    def test_pow(self):
        with pytest.raises(NotImplementedError):
            self.qarray**2

    def test_elpow(self):
        assert jnp.array_equal(self.qarray.elpow(2).to_jax(), self.data**2)
        assert jnp.array_equal(self.qarray.elpow(3).to_jax(), self.data**3)


@pytest.mark.run(order=TEST_SHORT)
def test_dense_qarray_axis_manipulation_methods():
    data = jnp.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4) * (1 + 1j)
    x = asqarray(data, dims=(4,))

    # batch-axis swapaxes preserves qarray structure
    y = x.swapaxes(0, 1)
    expected = jnp.swapaxes(data, 0, 1)
    assert y.shape == expected.shape
    assert y.dims == x.dims
    assert jnp.array_equal(y.to_jax(), expected)

    # final-two-axis swapaxes preserves qarray structure
    y = x.swapaxes(-1, -2)
    expected = jnp.swapaxes(data, -1, -2)
    assert y.shape == expected.shape
    assert y.dims == x.dims
    assert jnp.array_equal(y.to_jax(), expected)

    # moving batch axes preserves qarray structure
    y = x.moveaxis(0, 1)
    expected = jnp.moveaxis(data, 0, 1)
    assert y.shape == expected.shape
    assert y.dims == x.dims
    assert jnp.array_equal(y.to_jax(), expected)

    # inserting batch axes preserves qarray structure
    y = x.expand_dims(0)
    expected = jnp.expand_dims(data, 0)
    assert y.shape == expected.shape
    assert y.dims == x.dims
    assert jnp.array_equal(y.to_jax(), expected)

    y = x.expand_dims(2)
    expected = jnp.expand_dims(data, 2)
    assert y.shape == expected.shape
    assert y.dims == x.dims
    assert jnp.array_equal(y.to_jax(), expected)


@pytest.mark.run(order=TEST_SHORT)
def test_dense_qarray_axis_manipulation_raw_array_fallback():
    data = jnp.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4) * (1 + 1j)
    x = asqarray(data, dims=(4,))

    y = x.swapaxes(0, -1)
    expected = jnp.swapaxes(data, 0, -1)
    assert not hasattr(y, 'to_jax')
    assert y.shape == expected.shape
    assert jnp.array_equal(y, expected)

    y = x.moveaxis(-1, 0)
    expected = jnp.moveaxis(data, -1, 0)
    assert not hasattr(y, 'to_jax')
    assert y.shape == expected.shape
    assert jnp.array_equal(y, expected)

    y = x.expand_dims(-1)
    expected = jnp.expand_dims(data, -1)
    assert not hasattr(y, 'to_jax')
    assert y.shape == expected.shape
    assert jnp.array_equal(y, expected)
