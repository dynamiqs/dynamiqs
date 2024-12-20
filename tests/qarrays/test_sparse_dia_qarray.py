import warnings

import jax.numpy as jnp
import jax.random as jr
import pytest
from equinox import EquinoxRuntimeError

import dynamiqs as dq

from ..order import TEST_SHORT

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


@pytest.mark.run(order=TEST_SHORT)
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

        # matrix A
        make_dictA = lambda x: dict(
            simple=x,
            batch=dq.stack([x, 2 * x]),
            batch_broadcast=dq.stack([x, 2 * x]).reshape(2, 1, N, N),
        )

        sparseA = dq.SparseDIAQArray((N,), False, offsetsA, diagsA)
        denseA = sparseA.asdense()

        self.denseA = make_dictA(denseA)
        self.sparseA = make_dictA(sparseA)

        # matrix B
        make_dictB = lambda x: dict(
            simple=x,
            batch=dq.stack([x, 2 * x, 3 * x]),
            batch_broadcast=dq.stack([x, 2 * x, 3 * x]).reshape(1, 3, N, N),
        )

        sparseB = dq.SparseDIAQArray((N,), False, offsetsB, diagsB)
        denseB = sparseB.asdense()

        self.denseB = make_dictB(denseB)
        self.sparseB = make_dictB(sparseB)

        # scalar
        self.scalar = 2 + 2j
        self.bscalar = jnp.ones((2, 2, 1, 1), dtype=jnp.complex64)

    @pytest.mark.parametrize('kA', ['simple', 'batch', 'batch_broadcast'])
    def test_convert(self, kA):
        d = self.denseA[kA]
        assert _allclose(d.to_jax(), d.assparsedia().to_jax())

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_add(self, kA, kB):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]
        out_dense_dense = (dA + dB).to_jax()
        out_dia_dia = (sA + sB).to_jax()

        # check dia + dia
        assert _allclose(out_dense_dense, out_dia_dia)

        # check dia + dense
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            out_dia_dense = (sA + dB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dense)

        # check dense + dia
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            out_dense_dia = (dA + sB).to_jax()
        assert _allclose(out_dense_dense, out_dense_dia)

        # check addition with a scalar raises an error
        with pytest.raises(NotImplementedError):
            sA + self.scalar
        with pytest.raises(NotImplementedError):
            sA + self.bscalar

    @pytest.mark.parametrize('kA', ['simple', 'batch', 'batch_broadcast'])
    def test_scalaradd(self, kA):
        d, s = self.denseA[kA], self.sparseA[kA]

        assert _allclose(d.addscalar(self.scalar).to_jax(), d.to_jax() + self.scalar)
        assert _allclose(d.addscalar(self.bscalar).to_jax(), d.to_jax() + self.bscalar)
        assert _allclose(s.addscalar(self.scalar).to_jax(), s.to_jax() + self.scalar)
        assert _allclose(s.addscalar(self.bscalar).to_jax(), s.to_jax() + self.bscalar)

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_sub(self, kA, kB):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]
        out_dense_dense = (dA - dB).to_jax()
        out_dia_dia = (sA - sB).to_jax()

        # check dia - dia
        assert _allclose(out_dense_dense, out_dia_dia)

        # check dia - dense
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            out_dia_dense = (sA - dB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dense)

        # check dense - dia
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            out_dense_dia = (dA - sB).to_jax()
        assert _allclose(out_dense_dense, out_dense_dia)

        # check subtraction with a scalar raises an error
        with pytest.raises(NotImplementedError):
            sA - self.scalar
        with pytest.raises(NotImplementedError):
            sA - self.bscalar

    @pytest.mark.parametrize('kA', ['simple', 'batch', 'batch_broadcast'])
    def test_mul(self, kA):
        d, s = self.denseA[kA], self.sparseA[kA]

        assert _allclose((d * self.scalar).to_jax(), d.to_jax() * self.scalar)
        assert _allclose((d * self.bscalar).to_jax(), d.to_jax() * self.bscalar)
        assert _allclose((s * self.scalar).to_jax(), s.to_jax() * self.scalar)
        assert _allclose((s * self.bscalar).to_jax(), s.to_jax() * self.bscalar)

        # check multiplication with a qarray raises an error
        with pytest.raises(NotImplementedError):
            s * d
        with pytest.raises(NotImplementedError):
            s * s

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_elmul(self, kA, kB):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]
        out_dense_dense = dA.elmul(dB).to_jax()

        # check dia * dia
        out_dia_dia = sA.elmul(sB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dia)

        # check dia * dense
        out_dia_dense = sA.elmul(dB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dense)

        # check dense * dia
        out_dense_dia = dA.elmul(sB).to_jax()
        assert _allclose(out_dense_dense, out_dense_dia)

    @pytest.mark.parametrize(('kA', 'kB'), valid_operation_keys)
    def test_matmul(self, kA, kB):
        dA, sA = self.denseA[kA], self.sparseA[kA]
        dB, sB = self.denseB[kB], self.sparseB[kB]

        out_dense_dense = (dA @ dB).to_jax()

        out_dia_dia = (sA @ sB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dia)

        out_dia_dense = (sA @ dB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dense)

        out_dense_dia = (dA @ sB).to_jax()
        assert _allclose(out_dense_dense, out_dense_dia)

    def test_kronecker(self):
        dA, sA = self.denseA['simple'], self.sparseA['simple']
        dB, sB = self.denseB['simple'], self.sparseB['simple']

        out_dense_dense = (dA & dB).to_jax()

        out_dia_dia = (sA & sB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dia)

        out_dia_dense = (sA & dB).to_jax()
        assert _allclose(out_dense_dense, out_dia_dense)

    def test_outofbounds(self):
        # set up matrix
        N = 10
        diags = jr.normal(jr.key(42), (4, N))
        offsets = (-2, -1, 1, 3)

        # assert an error is raised
        error_str = 'must contain zeros outside the matrix bounds'
        with pytest.raises(EquinoxRuntimeError, match=error_str):
            dq.SparseDIAQArray((N,), False, offsets, diags)

    @pytest.mark.parametrize('k', ['simple', 'batch', 'batch_broadcast'])
    def test_elpow(self, k):
        d, s = self.denseA[k], self.sparseA[k]

        out_dense = d.elpow(3).to_jax()
        out_dia = s.elpow(3).to_jax()

        assert _allclose(out_dia, out_dense)

    @pytest.mark.parametrize('k', ['simple', 'batch', 'batch_broadcast'])
    def test_powm(self, k):
        d, s = self.denseA[k], self.sparseA[k]

        out_dense = d.powm(3).to_jax()
        out_dia = s.powm(3).to_jax()

        assert _allclose(out_dia, out_dense)


def _allclose(a, b, rtol=1e-05, atol=1e-08):
    return jnp.allclose(a, b, rtol=rtol, atol=atol)
