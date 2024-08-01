import random

import jax.numpy as jnp
import jax.random as jr
import pytest

import dynamiqs as dq


class TestSparseDIAQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        N = 10
        key = jr.key(42)
        keyA, keyB = jr.split(key)
        diagsA = jr.normal(keyA, (3, N))
        diagsB = jr.normal(keyB, (4, N))
        offsetsA = (-2, 0, 2)
        offsetsB = (-2, -1, 1, 3)

        # set out of bounds values to zero
        diagsA = diagsA.at[0, -2:].set(0)
        diagsA = diagsA.at[2, :2].set(0)
        diagsB = diagsB.at[0, -2:].set(0)
        diagsB = diagsB.at[1, -1:].set(0)
        diagsB = diagsB.at[2, :1].set(0)
        diagsB = diagsB.at[3, :3].set(0)

        self.sparseA = dq.SparseDIAQArray(diags=diagsA, offsets=offsetsA, dims=(N,))
        self.sparseB = dq.SparseDIAQArray(diags=diagsB, offsets=offsetsB, dims=(N,))

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
        out_dia_dia = (self.sparseA + self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA + self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (self.matrixA + self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_add_batch(self, rtol=1e-05, atol=1e-08):
        denseA = dq.stack([self.matrixA, 2 * self.matrixA])
        denseB = dq.stack([self.matrixB, 2 * self.matrixB])

        sparseA = dq.stack([self.sparseA, 2 * self.sparseA])
        sparseB = dq.stack([self.sparseB, 2 * self.sparseB])

        out_dense_dense = (denseA @ denseB).to_jax()
        out_dia_dia = (sparseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sparseA @ denseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (denseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_add_batch_broadcast(self, rtol=1e-05, atol=1e-08):
        # same as `test_matmul_batch` but with different batching axes
        n = self.matrixA.shape[-1]

        denseA = dq.stack([self.matrixA, 2 * self.matrixA])
        denseA = denseA.reshape(2, 1, n, n)

        denseB = dq.stack([self.matrixB, 2 * self.matrixB, 3 * self.matrixB])
        denseB = denseB.reshape(1, 3, n, n)

        sparseA = dq.stack([self.sparseA, 2 * self.sparseA])
        sparseA = sparseA.reshape(2, 1, n, n)

        sparseB = dq.stack([self.sparseB, 2 * self.sparseB, 3 * self.sparseB])
        sparseB = sparseB.reshape(1, 3, n, n)

        out_dense_dense = (denseA @ denseB).to_jax()
        out_dia_dia = (sparseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sparseA @ denseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (denseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_sub(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA - self.matrixB).to_jax()
        out_dia_dense = (self.sparseA - self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (self.matrixA - self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)
        out_dia_dia = (self.sparseA - self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

    def test_mul(self, rtol=1e-05, atol=1e-08):
        random_float = random.uniform(1.0, 10.0)
        out_dense_left = (random_float * self.matrixA).to_jax()
        out_dense_right = (self.matrixA * random_float).to_jax()
        assert jnp.allclose(out_dense_left, out_dense_right, rtol=rtol, atol=atol)

        out_dia_left = (random_float * self.sparseA).to_jax()
        assert jnp.allclose(out_dense_left, out_dia_left, rtol=rtol, atol=atol)

        out_dia_right = (self.sparseA * random_float).to_jax()
        assert jnp.allclose(out_dense_left, out_dia_right, rtol=rtol, atol=atol)

    def test_matmul(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA @ self.matrixB).to_jax()

        out_dia_dia = dq.to_dense(self.sparseA @ self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA @ self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (self.matrixA @ self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_matmul_batch(self, rtol=1e-05, atol=1e-08):
        denseA = dq.stack([self.matrixA, 2 * self.matrixA])
        denseB = dq.stack([self.matrixB, 2 * self.matrixB])

        sparseA = dq.stack([self.sparseA, 2 * self.sparseA])
        sparseB = dq.stack([self.sparseB, 2 * self.sparseB])

        out_dense_dense = (denseA @ denseB).to_jax()
        out_dia_dia = (sparseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sparseA @ denseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (denseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_matmul_batch_broadcast(self, rtol=1e-05, atol=1e-08):
        # same as `test_matmul_batch` but with different batching axes
        n = self.matrixA.shape[-1]

        denseA = dq.stack([self.matrixA, 2 * self.matrixA])
        denseA = denseA.reshape(2, 1, n, n)

        denseB = dq.stack([self.matrixB, 2 * self.matrixB, 3 * self.matrixB])
        denseB = denseB.reshape(1, 3, n, n)

        sparseA = dq.stack([self.sparseA, 2 * self.sparseA])
        sparseA = sparseA.reshape(2, 1, n, n)

        sparseB = dq.stack([self.sparseB, 2 * self.sparseB, 3 * self.sparseB])
        sparseB = sparseB.reshape(1, 3, n, n)

        out_dense_dense = (denseA @ denseB).to_jax()
        out_dia_dia = (sparseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sparseA @ denseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (denseA @ sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_kronecker(self, rtol=1e-05, atol=1e-08):
        out_dense_dense = (self.matrixA & self.matrixB).to_jax()

        out_dia_dia = (self.sparseA & self.sparseB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (self.sparseA & self.matrixB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

    def test_outofbounds(self):
        # set up matrix
        N = 10
        diags = jr.normal(jr.key(42), (4, N))
        offsets = (-2, -1, 1, 3)

        # assert an error is raised
        error_str = 'must contain zeros outside the matrix bounds'
        with pytest.raises(ValueError, match=error_str):
            dq.SparseDIAQArray(diags=diags, offsets=offsets, dims=(N,))
