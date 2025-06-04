import jax.numpy as jnp
import pytest

from dynamiqs import asqarray
from dynamiqs.qarrays.composite_qarray import CompositeQArray, _tensor_product_shape
from dynamiqs.qarrays.layout import composite

from ..order import TEST_SHORT


def test_tensor_product_shape():
    assert _tensor_product_shape([(2,), (3,)]) == (6,)
    assert _tensor_product_shape([(4, 4), (4, 4)]) == (16, 16)
    assert _tensor_product_shape([(4, 3, 2, 1), (5, 6)]) == (4, 3, 10, 6)
    assert _tensor_product_shape([(3, 2, 1), (4, 5), (6, 7)]) == (3, 48, 35)


@pytest.mark.run(order=TEST_SHORT)
class TestCompositeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.term_0_factor_0 = asqarray(
            jnp.arange(160, dtype=jnp.float32).reshape((10, 4, 4)), dims=(2, 2)
        )
        self.term_0_factor_1 = asqarray(
            asqarray(jnp.arange(81, dtype=jnp.float32).reshape((9, 9)), dims=(3, 3))
        )
        self.term_1_factor_0 = asqarray(
            jnp.arange(160, dtype=jnp.float32).reshape((10, 4, 4)), dims=(2, 2)
        )
        self.term_1_factor_1 = asqarray(
            asqarray(jnp.arange(81, dtype=jnp.float32).reshape((9, 9)), dims=(3, 3))
        )
        self.terms = [
            (self.term_0_factor_0, self.term_0_factor_1),
            (self.term_1_factor_0, self.term_1_factor_1),
        ]
        self.qarray = CompositeQArray((2, 3), False, self.terms)

    def test_dtype(self):
        assert self.qarray.dtype == jnp.float32

    def test_layout(self):
        assert self.qarray.layout is composite

    def test_shape(self):
        assert self.qarray.shape == (10, 36, 36)

    def test_mT(self):
        mT = self.qarray.mT
        assert isinstance(mT, CompositeQArray)

        assert len(mT.terms) == len(self.qarray.terms)

        for i, term in enumerate(mT.terms):
            assert len(term) == len(self.qarray.terms[i])

            for j, factor in enumerate(term):
                assert jnp.array_equal(
                    factor.to_jax(), self.qarray.terms[i][j].to_jax().mT
                )

    def test_conj(self):
        conj = self.qarray.conj()
        assert isinstance(conj, CompositeQArray)

        assert len(conj.terms) == len(self.qarray.terms)

        for i, term in enumerate(conj.terms):
            assert len(term) == len(self.qarray.terms[i])

            for j, factor in enumerate(term):
                assert jnp.array_equal(
                    factor.to_jax(), self.qarray.terms[i][j].to_jax().conj()
                )
