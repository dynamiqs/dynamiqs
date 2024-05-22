import jax.numpy as jnp
import pytest

import dynamiqs as dq


class TestSparseDIA:
    @pytest.fixture(autouse=True)
    def _setup(self):
        N = 4
        self.matrixA = dq.destroy(N)
        self.matrixB = dq.number(N)
        self.sparseA = dq.to_sparse(self.matrixA)
        self.sparseB = dq.to_sparse(self.matrixB)

    def test_matmul(self, rtol=1e-05, atol=1e-08):
        out_dia_dia = (self.sparseA @ self.sparseB).to_dense()
        out_dia_dense = self.sparseA @ self.matrixB
        out_dense_dia = self.matrixA @ self.sparseB
        out_dense_dense = self.matrixA @ self.matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)
