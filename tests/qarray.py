import jax.numpy as jnp
import pytest

from dynamiqs import QArray, dag, tensor


class TestQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = jnp.arange(16).reshape(4, 4) * (1 + 1j)
        self.qarray = QArray(self.data, dims=(2, 2))

        self.other = jnp.arange(16).reshape(4, 4) + 16
        self.qother = QArray(self.other, dims=(2, 2))

    def test_I(self):
        assert jnp.array_equal(self.qarray.I, jnp.eye(4))

    def test_dag(self):
        assert jnp.array_equal(self.qarray.dag().inner, dag(self.data))

    def test_ptrace(self):
        ptrace = self.qarray.ptrace((1,))
        expected_ptrace = jnp.array([[10, 12], [18, 20]]) * (1 + 1j)
        assert jnp.array_equal(ptrace.inner, expected_ptrace)
        assert ptrace.dims == (2,)

    def test_add(self):
        scalar = 2 + 2j

        assert jnp.array_equal((self.qarray + scalar).inner, self.data + scalar)
        assert jnp.array_equal((self.qarray + self.other).inner, self.data + self.other)
        assert jnp.array_equal(
            (self.qarray + self.qother).inner, self.data + self.other
        )

    def test_sub(self):
        scalar = 2 + 2j

        assert jnp.array_equal((self.qarray - scalar).inner, self.data - scalar)
        assert jnp.array_equal((self.qarray - self.other).inner, self.data - self.other)
        assert jnp.array_equal(
            (self.qarray - self.qother).inner, self.data - self.other
        )

    def test_mul(self):
        scalar = 2 + 2j

        assert jnp.array_equal((self.qarray * scalar).inner, self.data * scalar)
        assert jnp.array_equal((self.qarray * self.other).inner, self.data * self.other)
        assert jnp.array_equal(
            (self.qarray * self.qother).inner, self.data * self.other
        )

    def test_matmul(self):
        assert jnp.array_equal((self.qarray @ self.other).inner, self.data @ self.other)
        assert jnp.array_equal(
            (self.qarray @ self.qother).inner, self.data @ self.other
        )

    def test_rmatmul(self):
        assert jnp.array_equal((self.other @ self.qarray).inner, self.other @ self.data)
        assert jnp.array_equal(
            (self.qother @ self.qarray).inner, self.other @ self.data
        )

    def test_and(self):
        t = self.qarray & self.qother

        assert jnp.array_equal(t.inner, tensor(self.data, self.other))
        assert t.dims == (2, 2, 2, 2)

        other = jnp.arange(9).reshape(3, 3)
        qother = QArray(other)
        t = self.qarray & qother

        assert jnp.array_equal(t.inner, tensor(self.data, other))
        assert t.dims == (2, 2, 3)
