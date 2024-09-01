import jax.numpy as jnp
import jax.random as jr
import pytest

import dynamiqs as dq

# list of all pairs of operations that are legal
# i.e. all pairs except 'batch-batch'
# and 'batch-batch_broadcast' because of sizes
# (1, 2, n, n) and (3, n, n)
valid_operation_keys = [
    ('simple', 'simple'),
    ('simple', 'batch'),
    ('simple', 'batch_broadcast'),
    ('batch', 'simple'),
    ('batch_broadcast', 'simple'),
    ('batch_broadcast', 'batch'),
    ('batch_broadcast', 'batch_broadcast'),
]


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

        # = matrix A =
        make_dictA = lambda x: dict(
            simple=x,
            batch=dq.stack([x, 2 * x]),
            batch_broadcast=dq.stack([x, 2 * x]).reshape(2, 1, N, N),
        )

        sparseA = dq.SparseDIAQArray(diags=diagsA, offsets=offsetsA, dims=(N,))
        denseA = dq.to_dense(sparseA)

        self.denseA = make_dictA(denseA)
        self.sparseA = make_dictA(sparseA)

        # = matrix B =
        make_dictB = lambda x: dict(
            simple=x,
            batch=dq.stack([x, 2 * x, 3 * x]),
            batch_broadcast=dq.stack([x, 2 * x, 3 * x]).reshape(1, 3, N, N),
        )

        sparseB = dq.SparseDIAQArray(diags=diagsB, offsets=offsetsB, dims=(N,))
        denseB = dq.to_dense(sparseB)

        self.denseB = make_dictB(denseB)
        self.sparseB = make_dictB(sparseB)

    @pytest.mark.parametrize('kA', ['simple', 'batch', 'batch_broadcast'])
    def test_convert(self, kA, rtol=1e-05, atol=1e-08):
        assert jnp.allclose(
            self.denseA[kA].to_jax(),
            dq.to_sparse_dia(self.denseA[kA]).to_jax(),
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_add(self, kA, kB, rtol=1e-05, atol=1e-08):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]

        out_dense_dense = (dA + dB).to_jax()

        out_dia_dia = dq.to_dense(sA + sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sA + dB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (dA + sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_sub(self, kA, kB, rtol=1e-05, atol=1e-08):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]

        out_dense_dense = (dA - dB).to_jax()
        out_dia_dia = dq.to_dense(sA - sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sA - dB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (dA - sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_mul(self, kA, kB, rtol=1e-05, atol=1e-08):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]

        out_dense_dense = (dA * dB).to_jax()

        out_dia_dia = dq.to_dense(sA * sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sA * dB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (dA * sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_matmul(self, kA, kB, rtol=1e-05, atol=1e-08):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]

        out_dense_dense = (dA @ dB).to_jax()

        out_dia_dia = dq.to_dense(sA @ sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sA @ dB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)

        out_dense_dia = (dA @ sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_kronecker(self, rtol=1e-05, atol=1e-08):
        dA, sA = self.denseA['simple'], self.sparseA['simple']
        dB, sB = self.denseB['simple'], self.sparseB['simple']

        out_dense_dense = (dA & dB).to_jax()

        out_dia_dia = (sA & sB).to_jax()
        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)

        out_dia_dense = (sA & dB).to_jax()
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

    @pytest.mark.parametrize('k', ['simple', 'batch', 'batch_broadcast'])
    def test_pow(self, k, rtol=1e-05, atol=1e-08):
        d, s = self.denseA[k], self.sparseA[k]

        out_dense = (d**3).to_jax()
        out_dia = dq.to_dense(s**3).to_jax()

        assert jnp.allclose(out_dia, out_dense, rtol=rtol, atol=atol)

    @pytest.mark.parametrize('k', ['simple', 'batch', 'batch_broadcast'])
    def test_powm(self, k, rtol=1e-05, atol=1e-08):
        d, s = self.denseA[k], self.sparseA[k]

        out_dense = d.powm(3).to_jax()
        out_dia = s.powm(3).to_jax()

        assert jnp.allclose(out_dia, out_dense, rtol=rtol, atol=atol)
