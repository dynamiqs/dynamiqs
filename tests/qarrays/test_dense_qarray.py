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
        assert jnp.array_equal(self.qarray.dag().data, self.data.mT.conj())

    def test_ptrace(self):
        ptrace = self.qarray.ptrace(1)
        expected_ptrace = jnp.array([[10, 12], [18, 20]]) * (1 + 1j)
        assert jnp.array_equal(ptrace.data, expected_ptrace)
        assert ptrace.dims == (2,)

    def test_add(self):
        assert jnp.array_equal((self.qarray + self.other).data, self.data + self.other)
        assert jnp.array_equal((self.qarray + self.qother).data, self.data + self.other)
        with pytest.raises(NotImplementedError):
            self.qarray + self.scalar
        with pytest.raises(NotImplementedError):
            self.qarray + self.bscalar

    def test_radd(self):
        assert jnp.array_equal((self.other + self.qarray).data, self.other + self.data)
        assert jnp.array_equal((self.qother + self.qarray).data, self.other + self.data)
        with pytest.raises(NotImplementedError):
            self.scalar + self.qarray
        with pytest.raises(NotImplementedError):
            self.bscalar + self.qarray

    def test_scalaradd(self):
        assert jnp.array_equal(
            self.qarray.addscalar(self.scalar).data, self.data + self.scalar
        )
        assert jnp.array_equal(
            self.qarray.addscalar(self.bscalar).data, self.data + self.bscalar
        )

    def test_sub(self):
        assert jnp.array_equal((self.qarray - self.other).data, self.data - self.other)
        assert jnp.array_equal((self.qarray - self.qother).data, self.data - self.other)
        with pytest.raises(NotImplementedError):
            self.qarray - self.scalar
        with pytest.raises(NotImplementedError):
            self.qarray - self.bscalar

    def test_mul(self):
        assert jnp.array_equal(
            (self.qarray * self.scalar).data, self.data * self.scalar
        )
        assert jnp.array_equal(
            (self.qarray * self.bscalar).data, self.data * self.bscalar
        )

    def test_rmul(self):
        assert jnp.array_equal(
            (self.scalar * self.qarray).data, self.scalar * self.data
        )
        assert jnp.array_equal(
            (self.bscalar * self.qarray).data, self.bscalar * self.data
        )

    def test_elmul(self):
        assert jnp.array_equal(
            self.qarray.elmul(self.other).data, self.data * self.other
        )
        assert jnp.array_equal(
            self.qarray.elmul(self.qother).data, self.data * self.other
        )

    def test_matmul(self):
        assert jnp.array_equal((self.qarray @ self.other).data, self.data @ self.other)
        assert jnp.array_equal((self.qarray @ self.qother).data, self.data @ self.other)

    def test_rmatmul(self):
        assert jnp.array_equal((self.other @ self.qarray).data, self.other @ self.data)
        assert jnp.array_equal((self.qother @ self.qarray).data, self.other @ self.data)

    def test_and(self):
        t = self.qarray & self.qother

        assert jnp.array_equal(t.data, tensor(self.data, self.other).data)
        assert t.dims == (2, 2, 2, 2)

        other = jnp.arange(9).reshape(3, 3)
        qother = asqarray(other)
        t = self.qarray & qother

        assert jnp.array_equal(t.data, tensor(self.data, other).data)
        assert t.dims == (2, 2, 3)

    def test_powm(self):
        assert jnp.array_equal(self.qarray.powm(2).data, self.data @ self.data)
        assert jnp.array_equal(
            self.qarray.powm(3).data, self.data @ self.data @ self.data
        )

    def test_pow(self):
        with pytest.raises(NotImplementedError):
            self.qarray**2

    def test_elpow(self):
        assert jnp.array_equal(self.qarray.elpow(2).data, self.data**2)
        assert jnp.array_equal(self.qarray.elpow(3).data, self.data**3)
