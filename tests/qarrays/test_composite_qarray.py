import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import dynamiqs as dq
from dynamiqs.qarrays.composite_qarray import CompositeQArray, CompositeTerm
from dynamiqs.qarrays.layout import Layout
from dynamiqs.qarrays.materialized_qarray import MaterializedQArray

from ..order import TEST_SHORT

# === fixtures shared by the invalid-construction cases ===
# a term/qarray that is otherwise entirely valid, used to isolate exactly one bad
# field per case
_OP2 = dq.asqarray(np.eye(2, dtype=complex), dims=(2,))
_OP3 = dq.asqarray(np.eye(3, dtype=complex), dims=(3,))
_OP2_DIA = dq.asqarray(np.eye(2, dtype=complex), dims=(2,), layout=dq.dia)
_OP3_DIA = dq.asqarray(np.eye(3, dtype=complex), dims=(3,), layout=dq.dia)
_VALID_TERM = CompositeTerm(operators=(_OP2, _OP3))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('build', 'error'),
    [
        pytest.param(
            lambda: CompositeTerm(operators=[_OP2, _OP3]),
            TypeError,
            id='term-operators-not-a-tuple',
        ),
        pytest.param(
            lambda: CompositeTerm(operators=()), ValueError, id='term-operators-empty'
        ),
        pytest.param(
            lambda: CompositeTerm(operators=(jnp.eye(2), _OP3)),
            TypeError,
            id='term-operator-not-materialized-qarray',
        ),
        pytest.param(
            lambda: CompositeTerm(operators=(dq.fock(2, 0),)),
            ValueError,
            id='term-operator-not-square',
        ),
        pytest.param(
            lambda: CompositeTerm(operators=(_OP2, _OP3), coeff=jnp.ones(3)),
            ValueError,
            id='term-coeff-not-a-batched-scalar',
        ),
        pytest.param(
            lambda: CompositeTerm(operators=(_OP2, _OP3_DIA)),
            ValueError,
            id='term-mixed-operator-layouts',
        ),
        pytest.param(
            lambda: CompositeQArray((2,), (CompositeTerm(operators=(_OP2,)),)),
            ValueError,
            id='qarray-dims-single-subsystem',
        ),
        pytest.param(
            lambda: CompositeQArray((2, 3), [_VALID_TERM]),
            TypeError,
            id='qarray-terms-not-a-tuple',
        ),
        pytest.param(
            lambda: CompositeQArray((2, 3), ()), ValueError, id='qarray-terms-empty'
        ),
        pytest.param(
            lambda: CompositeQArray((3, 2), (_VALID_TERM,)),
            ValueError,
            id='qarray-dims-term-mismatch',
        ),
        pytest.param(
            lambda: CompositeQArray(
                (2, 3), (_VALID_TERM, CompositeTerm(operators=(_OP2_DIA, _OP3_DIA)))
            ),
            ValueError,
            id='qarray-mixed-term-layouts',
        ),
    ],
)
def test_rejects_invalid_construction(build, error):
    with pytest.raises(error):
        build()


@pytest.mark.run(order=TEST_SHORT)
def test_qarray_matches_independent_oracle():
    # two terms, each with 2 operators, with different batch shapes and
    # non-cancelling values: this exercises `CompositeTerm`'s own combination
    # logic (`reduce(&, operators)`) and the batch broadcast across terms at
    # once, so there is nothing left for a standalone term-level test to add.
    A0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B0 = np.diag([1.0, 2.0, 3.0])
    coeff0 = 2.0

    A1 = np.stack([np.array([[1.0, k + 1.0], [2.0, -k]]) for k in range(4)])
    B1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    coeff1 = 0.5 + 0.5j

    # independent oracle, computed from the raw arrays with numpy alone
    oracle = coeff0 * np.kron(A0, B0)[None] + coeff1 * np.stack(
        [np.kron(A1[k], B1) for k in range(4)]
    )

    term0 = CompositeTerm(
        operators=(dq.asqarray(A0, dims=(2,)), dq.asqarray(B0, dims=(3,))), coeff=coeff0
    )
    term1 = CompositeTerm(
        operators=(dq.asqarray(A1, dims=(2,)), dq.asqarray(B1, dims=(3,))), coeff=coeff1
    )
    c = CompositeQArray((2, 3), (term0, term1))

    assert c.shape == (4, 6, 6)
    assert c.ndim == 3
    assert c.layout is dq.dense
    assert jnp.iscomplexobj(jnp.zeros((), dtype=c.dtype))

    assert np.allclose(c.to_jax(), oracle, rtol=1e-5, atol=1e-5)
    assert np.allclose(c.to_numpy(), oracle, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.asarray(c), oracle, rtol=1e-5, atol=1e-5)
    assert np.allclose(c.mT.to_jax(), np.swapaxes(oracle, -1, -2), rtol=1e-5, atol=1e-5)

    assert c.devices() == set(jax.devices())

    blocked = c.block_until_ready()
    assert type(blocked) is CompositeQArray
    assert np.allclose(blocked.to_jax(), oracle, rtol=1e-5, atol=1e-5)

    qutip_list = c.to_qutip()
    assert len(qutip_list) == 4
    assert np.allclose(qutip_list[0].full(), oracle[0], rtol=1e-5, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_ndiags_is_lazy_and_guarded():
    # two dia terms with overlapping and non-overlapping offsets, so `ndiags`
    # must actually union and de-duplicate rather than just count one term
    A0 = dq.asqarray(np.diag([1.0, 2.0]), dims=(2,), layout=dq.dia)
    B0 = dq.asqarray(
        np.diag([1.0, 2.0, 3.0]) + np.diag([4.0, 5.0], k=1), dims=(3,), layout=dq.dia
    )
    A1 = dq.asqarray(np.eye(2), dims=(2,), layout=dq.dia)
    B1 = dq.asqarray(
        np.diag([1.0, 2.0, 3.0]) + np.diag([6.0, 7.0], k=-1), dims=(3,), layout=dq.dia
    )
    term0 = CompositeTerm(operators=(A0, B0))
    term1 = CompositeTerm(operators=(A1, B1))
    c_dia = CompositeQArray((2, 3), (term0, term1))

    assert c_dia.ndiags == 3
    assert c_dia.ndiags == c_dia._materialize().ndiags

    c_dense = CompositeQArray((2, 3), (_VALID_TERM,))
    with pytest.raises(AttributeError):
        _ = c_dense.ndiags


@pytest.mark.run(order=TEST_SHORT)
def test_layout_conversion_stays_lazy():
    A0 = dq.asqarray(np.diag([1.0, 2.0]), dims=(2,), layout=dq.dia)
    B0 = dq.asqarray(np.diag([1.0, 2.0, 3.0]), dims=(3,), layout=dq.dia)
    c_dia = CompositeQArray((2, 3), (CompositeTerm(operators=(A0, B0)),))

    dense = c_dia.asdense()
    assert type(dense) is CompositeQArray
    assert dense.layout is dq.dense
    assert jnp.allclose(dense.to_jax(), c_dia.to_jax())

    c_dense = CompositeQArray((2, 3), (_VALID_TERM,))
    sparse = c_dense.assparsedia()
    assert type(sparse) is CompositeQArray
    assert sparse.layout is dq.dia
    assert jnp.allclose(sparse.to_jax(), c_dense.to_jax())

    # explicit offsets designate diagonals of the full matrix, which do not
    # decompose into per-operator offsets: this branch must materialize
    materialized = c_dense._materialize()
    offsets = materialized.assparsedia().data.offsets
    explicit = c_dense.assparsedia(offsets=offsets)
    assert type(explicit) is MaterializedQArray
    assert jnp.allclose(explicit.to_jax(), materialized.to_jax())


@pytest.mark.run(order=TEST_SHORT)
def test_repr_does_not_materialize(monkeypatch):
    c = CompositeQArray(
        (2, 3),
        (
            CompositeTerm(operators=(_OP2, _OP3), coeff=2.0),
            CompositeTerm(operators=(_OP2, _OP3), coeff=3.0),
        ),
    )

    def _boom(self):
        raise AssertionError('`__repr__` must not materialize')

    monkeypatch.setattr(CompositeQArray, '_materialize', _boom)

    text = repr(c)
    assert 'CompositeQArray' in text
    assert 'n_terms=2' in text
    assert str(c.dims) in text
    assert str(c.shape) in text


def _batched_composite(layout: Layout) -> tuple[CompositeQArray, np.ndarray]:
    # two terms with different original batch shapes, so every batch-axis method
    # exercises `_aligned`; overall shape (1, 4, 6, 6)
    A0 = dq.asqarray(np.diag([1.0, 2.0]), dims=(2,), layout=layout)
    B0 = dq.asqarray(np.diag([1.0, 2.0, 3.0]), dims=(3,), layout=layout)
    term0 = CompositeTerm(operators=(A0, B0), coeff=2.0)

    A1_data = np.stack([np.diag([1.0, k + 2.0]) for k in range(4)])
    A1 = dq.asqarray(A1_data, dims=(2,), layout=layout)
    B1_data = np.diag([2.0, 1.0, 3.0])
    B1 = dq.asqarray(B1_data, dims=(3,), layout=layout)
    coeff1 = jnp.arange(1.0, 5.0).reshape(1, 4, 1, 1)
    term1 = CompositeTerm(operators=(A1, B1), coeff=coeff1)

    c = CompositeQArray((2, 3), (term0, term1))

    oracle0 = 2.0 * np.kron(np.diag([1.0, 2.0]), np.diag([1.0, 2.0, 3.0]))
    oracle1 = np.stack([(k + 1) * np.kron(A1_data[k], B1_data) for k in range(4)])
    oracle = oracle0[None, None] + oracle1[None]
    return c, oracle


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_reshape_matches_oracle(layout):
    c, oracle = _batched_composite(layout)

    reshaped = c.reshape(4, 6, 6)

    assert type(reshaped) is CompositeQArray
    assert reshaped.layout is layout
    assert reshaped.shape == (4, 6, 6)
    assert jnp.allclose(reshaped.to_jax(), oracle.reshape(4, 6, 6))


@pytest.mark.run(order=TEST_SHORT)
def test_reshape_rejects_matrix_axis_change():
    c, _ = _batched_composite(dq.dense)

    with pytest.raises(ValueError, match='last two dimensions'):
        c.reshape(4, 4, 9)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_broadcast_to_matches_oracle(layout):
    c, oracle = _batched_composite(layout)

    broadcasted = c.broadcast_to(3, 1, 4, 6, 6)

    assert type(broadcasted) is CompositeQArray
    assert broadcasted.layout is layout
    assert broadcasted.shape == (3, 1, 4, 6, 6)
    assert jnp.allclose(broadcasted.to_jax(), np.broadcast_to(oracle, (3, 1, 4, 6, 6)))


@pytest.mark.run(order=TEST_SHORT)
def test_broadcast_to_rejects_matrix_axis_change():
    c, _ = _batched_composite(dq.dense)

    with pytest.raises(ValueError, match='last two dimensions'):
        c.broadcast_to(1, 4, 7, 7)


@pytest.mark.run(order=TEST_SHORT)
def test_swapaxes_matches_oracle():
    c, oracle = _batched_composite(dq.dense)

    swapped = c.swapaxes(0, 1)

    assert type(swapped) is CompositeQArray
    assert swapped.shape == (4, 1, 6, 6)
    assert jnp.allclose(swapped.to_jax(), np.swapaxes(oracle, 0, 1))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('source', 'destination'),
    [
        pytest.param(0, 1, id='single-axis'),
        pytest.param((0, 1), (1, 0), id='multi-axis'),
    ],
)
def test_moveaxis_matches_oracle(source, destination):
    c, oracle = _batched_composite(dq.dense)

    moved = c.moveaxis(source, destination)
    expected = np.moveaxis(oracle, source, destination)

    assert type(moved) is CompositeQArray
    assert moved.shape == expected.shape
    assert jnp.allclose(moved.to_jax(), expected)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    'axis', [pytest.param(0, id='single-axis'), pytest.param((0, 2), id='multi-axis')]
)
def test_expand_dims_matches_oracle(axis):
    c, oracle = _batched_composite(dq.dense)

    expanded = c.expand_dims(axis)
    expected = np.expand_dims(oracle, axis)

    assert type(expanded) is CompositeQArray
    assert expanded.shape == expected.shape
    assert jnp.allclose(expanded.to_jax(), expected)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_squeeze_batch_axis_stays_lazy(layout):
    c, oracle = _batched_composite(layout)

    squeezed = c.squeeze(0)

    assert type(squeezed) is CompositeQArray
    assert squeezed.layout is layout
    assert squeezed.shape == (4, 6, 6)
    assert jnp.allclose(squeezed.to_jax(), np.squeeze(oracle, 0))


@pytest.mark.run(order=TEST_SHORT)
def test_squeeze_matrix_axis_materializes():
    A = dq.asqarray(np.array([[2.0]]), dims=(1,))
    B = dq.asqarray(np.array([[3.0]]), dims=(1,))
    c = CompositeQArray((1, 1), (CompositeTerm(operators=(A, B)),))

    squeezed = c.squeeze(-1)

    assert not isinstance(squeezed, CompositeQArray)
    assert jnp.allclose(jnp.asarray(squeezed), np.squeeze(np.array([[6.0]]), axis=-1))


def _squeeze_none_batch_axis_case() -> tuple[CompositeQArray, np.ndarray]:
    # matrix dims (6, 6) are not size 1, so an unqualified squeeze() should
    # only remove the size-1 batch axis and stay lazy
    return _batched_composite(dq.dense)


def _squeeze_none_matrix_axis_case() -> tuple[CompositeQArray, np.ndarray]:
    # matrix dims are (1, 1), so an unqualified squeeze() must materialize
    A = dq.asqarray(np.array([[2.0]]), dims=(1,))
    B = dq.asqarray(np.array([[3.0]]), dims=(1,))
    c = CompositeQArray((1, 1), (CompositeTerm(operators=(A, B)),))
    return c, np.array([[6.0]])


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('build', 'stays_lazy'),
    [
        pytest.param(_squeeze_none_batch_axis_case, True, id='batch-axis-only'),
        pytest.param(_squeeze_none_matrix_axis_case, False, id='matrix-dims-size-one'),
    ],
)
def test_squeeze_none_axis_materializes_only_when_matrix_dims_are_one(
    build, stays_lazy
):
    c, oracle = build()

    squeezed = c.squeeze()

    assert isinstance(squeezed, CompositeQArray) is stays_lazy
    assert np.allclose(np.asarray(squeezed), np.squeeze(oracle))


@pytest.mark.run(order=TEST_SHORT)
def test_sum_batch_axis_matches_oracle():
    c, oracle = _batched_composite(dq.dense)

    total = c.sum(axis=1)

    assert type(total) is MaterializedQArray
    assert total.shape == (1, 6, 6)
    assert jnp.allclose(total.to_jax(), oracle.sum(axis=1))


@pytest.mark.run(order=TEST_SHORT)
def test_sum_all_axes_returns_scalar_array():
    c, oracle = _batched_composite(dq.dense)

    total = c.sum(axis=None)

    assert isinstance(total, Array)
    assert jnp.allclose(total, oracle.sum())


@pytest.mark.run(order=TEST_SHORT)
def test_reshape_unchecked_materializes():
    c, oracle = _batched_composite(dq.dense)

    reshaped = c._reshape_unchecked(1, 4, 36)

    assert type(reshaped) is MaterializedQArray
    assert reshaped.shape == (1, 4, 36)
    assert jnp.allclose(reshaped.to_jax(), oracle.reshape(1, 4, 36))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
def test_getitem_batch_axis_stays_lazy(layout):
    c, oracle = _batched_composite(layout)

    key = (0, slice(1, 3))
    indexed = c[key]

    assert type(indexed) is CompositeQArray
    assert indexed.layout is layout
    assert indexed.shape == (2, 6, 6)
    assert jnp.allclose(indexed.to_jax(), oracle[key])


@pytest.mark.run(order=TEST_SHORT)
def test_getitem_matrix_axis_materializes():
    c, oracle = _batched_composite(dq.dense)

    key = (Ellipsis, 0)
    indexed = c[key]

    assert not isinstance(indexed, CompositeQArray)
    assert jnp.allclose(jnp.asarray(indexed), oracle[key])
