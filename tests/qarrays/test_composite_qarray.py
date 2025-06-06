from math import prod

import jax.numpy as jnp
import pytest

from dynamiqs import DenseQArray, asqarray, eye_like, sigmax, sigmay, sigmaz
from dynamiqs.qarrays.composite_qarray import CompositeQArray
from dynamiqs.qarrays.layout import Layout, dia

from ..order import TEST_SHORT


def _get_qarray(
    dims: tuple[int, ...],
    bshape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.float32,
    layout: Layout | None = None,
):
    shape = bshape + tuple(d**2 for d in dims)
    return asqarray(
        jnp.arange(prod(bshape) * prod(dims) ** 2, dtype=dtype).reshape(shape),
        dims=dims,
        layout=layout,
    )


@pytest.mark.run(order=TEST_SHORT)
class TestCompositeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # (\sigma_x ⊗ \sigma_y) + (\sigma_z ⊗ \sigma_x)
        self.qarray = CompositeQArray(
            (2, 2), False, [(sigmax(), sigmay()), (sigmaz(), sigmax())]
        )
        self.qarray_44 = _get_qarray(dims=(2, 2))
        self.qarray_44_dia = _get_qarray(dims=(2, 2), layout=dia)
        self.qarray_44_int = _get_qarray(dims=(2, 2), dtype=jnp.int32)
        self.qarray_244 = _get_qarray(dims=(2, 2), bshape=(2,))
        self.qarray_344 = _get_qarray(dims=(2, 2), bshape=(3,))

    def test_init_no_terms(self):
        with pytest.raises(ValueError, match='at least one term'):
            CompositeQArray((0,), False, [])

    def test_init_no_factor(self):
        with pytest.raises(ValueError, match='at least one factor'):
            CompositeQArray((0,), False, [()])

    def test_init_different_dtype(self):
        with pytest.raises(ValueError, match='same dtype'):
            CompositeQArray((2, 2), False, [(self.qarray_44, self.qarray_44_int)])

    def test_init_no_broadcast(self):
        with pytest.raises(ValueError, match='broadcastable shapes'):
            CompositeQArray((2, 2), False, [(self.qarray_244, self.qarray_344)])

        with pytest.raises(ValueError, match='broadcastable shapes'):
            CompositeQArray((2, 2), False, [(self.qarray_244,), (self.qarray_344,)])

    def test_dtype(self):
        assert CompositeQArray((2,), False, [(self.qarray_44_int,)]).dtype == jnp.int32

    def test_shape(self):
        assert CompositeQArray(
            (2, 2), False, [(self.qarray_44, self.qarray_244)]
        ).shape == (2, 16, 16)

    def test_mT(self):
        assert jnp.array_equal(
            self.qarray.mT.asdense().data,
            jnp.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, -1], [-1j, 0, -1, 0]]),
        )

    def test_conj(self):
        assert jnp.array_equal(
            self.qarray.conj().asdense().data,
            jnp.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, -1], [-1j, 0, -1, 0]]),
        )

    def test_trace(self):
        qarray = CompositeQArray((2, 2), False, [(self.qarray_44, self.qarray_244)])
        assert qarray.trace().shape == (2,)

    def test_isherm(self):
        assert not self.qarray_44.isherm()
        assert not CompositeQArray((2, 2), False, [(self.qarray_44,)]).isherm()

        assert self.qarray.isherm()

    def test_asdense(self):
        assert jnp.array_equal(
            self.qarray.asdense().data,
            jnp.array([[0, 1, 0, -1j], [1, 0, 1j, 0], [0, -1j, 0, -1], [1j, 0, -1, 0]]),
        )

    def test_assparsedia(self):
        assert jnp.array_equal(
            self.qarray.assparsedia().asdense().data,
            jnp.array([[0, 1, 0, -1j], [1, 0, 1j, 0], [0, -1j, 0, -1], [1j, 0, -1, 0]]),
        )

    def test__mul__(self):
        actual = self.qarray.__mul__(5)
        assert isinstance(actual, CompositeQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array([[0, 5, 0, -5j], [5, 0, 5j, 0], [0, -5j, 0, -5], [5j, 0, -5, 0]]),
        )

    def test__add__composite(self):
        actual = self.qarray.__add__(self.qarray)
        assert isinstance(actual, CompositeQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array([[0, 2, 0, -2j], [2, 0, 2j, 0], [0, -2j, 0, -2], [2j, 0, -2, 0]]),
        )

    def test__add__dense(self):
        actual = self.qarray.__add__(self.qarray.asdense())
        assert isinstance(actual, CompositeQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array([[0, 2, 0, -2j], [2, 0, 2j, 0], [0, -2j, 0, -2], [2j, 0, -2, 0]]),
        )

    def test__add__dia(self):
        actual = self.qarray.__add__(eye_like(self.qarray, layout=dia))
        assert isinstance(actual, CompositeQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array([[1, 1, 0, -1j], [1, 1, 1j, 0], [0, -1j, 1, -1], [1j, 0, -1, 1]]),
        )

    def test__add__array(self):
        actual = self.qarray.__add__(jnp.array([1, 2, 3, 4], dtype=jnp.complex64))
        assert isinstance(actual, DenseQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array(
                [
                    [1, 3, 3, 4 - 1j],
                    [2, 2, 3 + 1j, 4],
                    [1, 2 - 1j, 3, 3],
                    [1 + 1j, 2, 2, 4],
                ]
            ),
        )

    def test__matmul__(self):
        actual = self.qarray @ jnp.array([0, 0, 1, 0])
        assert isinstance(actual, DenseQArray)
        assert actual.shape == (4,)
        assert jnp.array_equal(actual.asdense().data, jnp.array([0, 1j, 0, -1]))

    def test__and__(self):
        # \sigma_x + \sigma_z
        qarray = CompositeQArray((2,), False, [(sigmax(),), (sigmaz(),)])

        # (\sigma_x ⊗ \sigma_y) + (\sigma_z ⊗ \sigma_y)
        actual = qarray & sigmay()
        assert isinstance(actual, CompositeQArray)

        assert jnp.array_equal(
            actual.asdense().data,
            jnp.array(
                [[0, -1j, 0, -1j], [1j, 0, 1j, 0], [0, -1j, 0, 1j], [1j, 0, -1j, 0]]
            ),
        )
        assert actual.dims == (2, 2)

    def test_addscalar(self):
        actual = self.qarray.addscalar(4)
        assert isinstance(actual, DenseQArray)
        assert jnp.array_equal(
            actual.data,
            jnp.array(
                [
                    [4, 5, 4, 4 - 1j],
                    [5, 4, 4 + 1j, 4],
                    [4, 4 - 1j, 4, 3],
                    [4 + 1j, 4, 3, 4],
                ]
            ),
        )

    def test_elmul(self):
        actual = self.qarray.elmul(self.qarray)
        assert isinstance(actual, DenseQArray)
        assert jnp.array_equal(
            actual.data,
            jnp.array([[0, 1, 0, -1], [1, 0, -1, 0], [0, -1, 0, 1], [-1, 0, 1, 0]]),
        )

    def test_elpow(self):
        actual = self.qarray.elpow(2)
        assert isinstance(actual, DenseQArray)
        assert jnp.array_equal(
            actual.data,
            jnp.array([[0, 1, 0, -1], [1, 0, -1, 0], [0, -1, 0, 1], [-1, 0, 1, 0]]),
        )
