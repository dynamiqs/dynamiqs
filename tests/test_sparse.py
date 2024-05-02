import jax
import jax.numpy as jnp

from dynamiqs.sparse import *


def assert_equal(arr1, arr2, arr3, arr4, rtol=1e-05, atol=1e-08):
    close12 = jnp.allclose(arr1, arr2, rtol=rtol, atol=atol)
    close13 = jnp.allclose(arr1, arr3, rtol=rtol, atol=atol)
    close14 = jnp.allclose(arr1, arr4, rtol=rtol, atol=atol)

    return close12 & close13 & close14


class TestSparseDIA:
    def test_matmul(self):
        N = 4
        keyA, keyB = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
        matrixA, matrixB = (
            jax.random.normal(keyA, (N, N)),
            jax.random.normal(keyB, (N, N)),
        )

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = sparseA @ sparseB
        out_dia_dense = sparseA @ matrixB
        out_dense_dia = matrixA @ sparseB
        out_dense_dense = matrixA @ matrixB

        assert assert_equal(
            out_dia_dia.to_dense(), out_dia_dense, out_dense_dia, out_dense_dense
        )

    def test_add(self):
        N = 4
        keyA, keyB = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
        matrixA, matrixB = (
            jax.random.normal(keyA, (N, N)),
            jax.random.normal(keyB, (N, N)),
        )

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = sparseA + sparseB
        out_dia_dense = sparseA + matrixB
        out_dense_dia = matrixA + sparseB
        out_dense_dense = matrixA + matrixB

        assert assert_equal(
            out_dia_dia.to_dense(), out_dia_dense, out_dense_dia, out_dense_dense
        )

    def test_transform(self, matrix):
        assert jnp.allclose(matrix, to_sparse(matrix).to_dense())
