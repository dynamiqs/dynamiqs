import random

import jax.numpy as jnp
import pytest

import dynamiqs as dq


class TestSparseDIAQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        N = 10
        num_diags = 3
        diags = jnp.arange(num_diags * N).reshape(num_diags, N)
        offsets = tuple(range(-(num_diags // 2), (num_diags // 2) + 1))

        self.sparseA = dq.SparseDIAQArray(diags=diags, offsets=offsets, dims=(N, N))
        self.sparseB = dq.SparseDIAQArray(diags=diags, offsets=offsets, dims=(N, N))

        self.matrixA = dq.to_dense(self.sparseA)
        self.matrixB = dq.to_dense(self.sparseB)

    def test_convert(self, rtol=1e-05, atol=1e-08):
        assert jnp.allclose(
            self.matrixA.to_jax(),
            dq.to_sparse_dia(self.matrixA).to_jax(),
            rtol=rtol,
            atol=atol,
        )

    def test_add(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA + self.matrixB).to_jax()

        out_dia_dia = dq.to_dense(self.sparseA + self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA + self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (self.matrixA + self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_sub(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA - self.matrixB).to_jax()

        out_dia_dense = self.sparseA - self.matrixB
        assert jnp.allclose(
            out_dense_dense, out_dia_dense.to_jax(), rtol=rtol, atol=atol
        )

        out_dense_dia = self.matrixA - self.sparseB
        assert jnp.allclose(
            out_dense_dense, out_dense_dia.to_jax(), rtol=rtol, atol=atol
        )
        out_dia_dia = dq.to_dense(self.sparseA - self.sparseB)
        assert jnp.allclose(out_dense_dense, out_dia_dia.to_jax(), rtol=rtol, atol=atol)

    def test_mul(self, rtol=1e-05, atol=1e-08):
        random_float = random.uniform(1.0, 10.0)

        out_dense_left = (random_float * self.matrixA).to_jax()

        out_dense_right = (self.matrixA * random_float).to_jax()
        assert jnp.allclose(out_dense_left, out_dense_right, rtol=rtol, atol=atol)

        out_dia_left = dq.to_dense(random_float * self.sparseA).to_jax()
        assert jnp.allclose(out_dense_left, out_dia_left, rtol=rtol, atol=atol)

        out_dia_right = dq.to_dense(self.sparseA * random_float).to_jax()
        assert jnp.allclose(out_dense_left, out_dia_right, rtol=rtol, atol=atol)

    def test_matmul(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA @ self.matrixB).to_jax()

        out_dia_dia = dq.to_dense(self.sparseA @ self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA @ self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (self.matrixA @ self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_kronecker(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA & self.matrixB).to_jax()

        out_dia_dia = (self.sparseA & self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA & self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
