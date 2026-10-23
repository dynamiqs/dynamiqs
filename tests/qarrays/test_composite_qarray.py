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
            lambda: CompositeTerm(
                operators=(_OP2.broadcast_to(2, 2, 2), _OP3), coeff=jnp.ones(3)
            ),
            ValueError,
            id='term-coeff-batch-shape-mismatch',
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
    A0 = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
    B0 = np.diag([1.0, 2.0, 3.0])  # (3,3)
    coeff0 = 2.0

    A1 = np.stack([np.array([[1.0, k + 1.0], [2.0, -k]]) for k in range(4)])  # (4,2,2)
    B1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # (3,3)
    coeff1 = (0.5 + 0.5j) * np.arange(1.0, 5.0)  # (4,)

    # independent oracle, computed from the raw arrays with numpy alone
    # (1,6,6) broadcast (4,6,6) → (4,6,6)
    oracle = coeff0 * np.kron(A0, B0)[None] + coeff1[:, None, None] * np.stack(
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
    A1 = dq.asqarray(np.eye(2), dims=(2,), layout=dq.dia)  # offsets (0,)
    B1 = dq.asqarray(
        np.diag([1.0, 2.0, 3.0]) + np.diag([6.0, 7.0], k=-1), dims=(3,), layout=dq.dia
    )  # offsets (-1, 0)
    term0 = CompositeTerm(operators=(A0, B0))
    term1 = CompositeTerm(operators=(A1, B1))
    # expected offsets {-1, 0, 1}, offset 0 appears once
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

    # off-diagonal operators, so that wrong per-operator offsets would drop entries
    A = dq.asqarray(np.array([[0.0, 1.0], [1.0, 0.0]]), dims=(2,))
    B = dq.asqarray(np.diag([1.0, 2.0, 3.0]) + np.diag([4.0, 5.0], k=-1), dims=(3,))
    c_off = CompositeQArray((2, 3), (CompositeTerm(operators=(A, B)),))
    offsets = (((-1, 1), (-1, 0)),)

    # static per-operator offsets stay lazy under `jax.jit`, where finding the
    # offsets from data is impossible
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda c: c.assparsedia())(c_off)
    explicit = jax.jit(lambda c: c.assparsedia(offsets=offsets))(c_off)
    assert type(explicit) is CompositeQArray
    assert explicit.layout is dq.dia
    assert jnp.allclose(explicit.to_jax(), c_off.to_jax())

    with pytest.raises(ValueError, match='one entry per operator'):
        c_off.assparsedia(offsets=(((-1, 1),),))
    with pytest.raises(ValueError, match='one entry per term'):
        c_off.assparsedia(offsets=(None, None))
    # diagonals of the full matrix do not decompose into per-operator offsets
    with pytest.raises(ValueError, match='diagonals of the full matrix'):
        c_off.assparsedia(offsets=c_off._materialize().assparsedia().data.offsets)


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
    # must first broadcast them to the full batch shape; overall shape (1, 4, 6, 6)
    A0 = dq.asqarray(np.diag([1.0, 2.0]), dims=(2,), layout=layout)
    B0 = dq.asqarray(np.diag([1.0, 2.0, 3.0]), dims=(3,), layout=layout)
    term0 = CompositeTerm(operators=(A0, B0), coeff=2.0)

    A1_data = np.stack([np.diag([1.0, k + 2.0]) for k in range(4)])
    A1 = dq.asqarray(A1_data, dims=(2,), layout=layout)
    B1_data = np.diag([2.0, 1.0, 3.0])
    B1 = dq.asqarray(B1_data, dims=(3,), layout=layout)
    coeff1 = jnp.arange(1.0, 5.0).reshape(1, 4)
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

    # both `A1` and the coefficient of the second term vary along axis 1, so
    # Σ_k c_k·A1_k ⊗ B1 is not a single term: the sum materializes
    total = c.sum(axis=1)

    assert type(total) is MaterializedQArray
    assert total.shape == (1, 6, 6)
    assert jnp.allclose(total.to_jax(), oracle.sum(axis=1))

    # with a scalar coefficient only `A1` varies, so the sum factorizes and stays
    # lazy, and the first term, constant along axis 1, is counted 4 times
    term1 = CompositeTerm(operators=c.terms[1].operators, coeff=3.0)
    c_lazy = CompositeQArray((2, 3), (c.terms[0], term1))
    total = c_lazy.sum(axis=-3)

    assert type(total) is CompositeQArray
    assert total.shape == (6, 6)
    assert jnp.allclose(total.to_jax(), c_lazy.to_jax().sum(axis=-3), atol=1e-5)


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


# === unary algebra ===
# `_batched_composite` is real and diagonal, which most wrong formulas survive, so
# the builders below use complex, non-normal and non-Hermitian data


def _random(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _general_composite() -> tuple[CompositeQArray, np.ndarray]:
    # two terms of batch shapes () and (3,), the second with a batched coefficient
    rng = np.random.default_rng(42)
    A0, B0, coeff0 = _random(rng, 2, 2), _random(rng, 3, 3), 1.5 - 0.5j
    A1, B1, coeff1 = _random(rng, 3, 2, 2), _random(rng, 3, 3), _random(rng, 3)
    term0 = CompositeTerm(
        (dq.asqarray(A0, dims=(2,)), dq.asqarray(B0, dims=(3,))), coeff0
    )
    term1 = CompositeTerm(
        (dq.asqarray(A1, dims=(2,)), dq.asqarray(B1, dims=(3,))), coeff1
    )
    c = CompositeQArray((2, 3), (term0, term1))
    # np.kron of shapes (3, 2, 2) and (1, 3, 3) is the batched Kronecker product
    oracle = coeff0 * np.kron(A0, B0) + coeff1[:, None, None] * np.kron(A1, B1[None])
    return c, oracle


def _single_term_composite() -> tuple[CompositeQArray, np.ndarray]:
    # non-normal factors with distinct eigenvalues, the second one batched only
    rng = np.random.default_rng(42)
    A, B, coeff = np.array([[1.0, 2.0], [0.0, 3.0]]), _random(rng, 3, 3, 3), 0.5j
    term = CompositeTerm((dq.asqarray(A, dims=(2,)), dq.asqarray(B, dims=(3,))), coeff)
    return CompositeQArray((2, 3), (term,)), coeff * np.kron(A[None], B)


def _hermitian_term_composite() -> tuple[CompositeQArray, np.ndarray]:
    # (iA) ⊗ (iB) = -A ⊗ B with A and B Hermitian: Hermitian, with factors that
    # are not, like iY ⊗ iY
    rng = np.random.default_rng(42)
    A, B = _random(rng, 2, 2), _random(rng, 3, 3)
    iA, iB = 1j * (A + A.conj().T), 1j * (B + B.conj().T)
    term = CompositeTerm((dq.asqarray(iA, dims=(2,)), dq.asqarray(iB, dims=(3,))))
    return CompositeQArray((2, 3), (term,)), np.kron(iA, iB)


def _hermitian_sum_composite() -> tuple[CompositeQArray, np.ndarray]:
    # A ⊗ A† + A† ⊗ A: Hermitian by cancellation between terms
    rng = np.random.default_rng(42)
    A = _random(rng, 2, 2)
    Ad = A.conj().T
    term0 = CompositeTerm((dq.asqarray(A, dims=(2,)), dq.asqarray(Ad, dims=(2,))))
    term1 = CompositeTerm((dq.asqarray(Ad, dims=(2,)), dq.asqarray(A, dims=(2,))))
    c = CompositeQArray((2, 2), (term0, term1))
    return c, np.kron(A, Ad) + np.kron(Ad, A)


def _nearly_hermitian_composite() -> tuple[CompositeQArray, np.ndarray]:
    # A is Hermitian up to `atol=1e-8` but A ⊗ B is not, since B scales the error
    eps, scale = 5e-9, 100.0
    A, B = np.array([[1.0, eps], [0.0, 1.0]]), scale * np.eye(2)
    term = CompositeTerm((dq.asqarray(A, dims=(2,)), dq.asqarray(B, dims=(2,))))
    return CompositeQArray((2, 2), (term,)), np.kron(A, B)


def _ptrace_composite(layout: Layout) -> tuple[CompositeQArray, np.ndarray]:
    # dims (2, 3, 2), the second term with an operator spanning subsystems 0 and 1
    rng = np.random.default_rng(42)
    A0, B0, C0 = _random(rng, 2, 2), _random(rng, 3, 3), _random(rng, 2, 2)
    AB1, C1 = _random(rng, 6, 6), _random(rng, 2, 2)
    term0 = CompositeTerm(
        (
            dq.asqarray(A0, dims=(2,), layout=layout),
            dq.asqarray(B0, dims=(3,), layout=layout),
            dq.asqarray(C0, dims=(2,), layout=layout),
        ),
        1.5,
    )
    term1 = CompositeTerm(
        (
            dq.asqarray(AB1, dims=(2, 3), layout=layout),
            dq.asqarray(C1, dims=(2,), layout=layout),
        ),
        0.5j,
    )
    c = CompositeQArray((2, 3, 2), (term0, term1))
    oracle = 1.5 * np.kron(np.kron(A0, B0), C0) + 0.5j * np.kron(AB1, C1)
    return c, oracle


def _forbid_materialize(monkeypatch, message: str):
    def _boom(self):
        raise AssertionError(message)

    monkeypatch.setattr(CompositeQArray, '_materialize', _boom)
    monkeypatch.setattr(CompositeTerm, '_materialize', _boom)


@pytest.mark.run(order=TEST_SHORT)
def test_conj_stays_lazy():
    c, oracle = _general_composite()

    conjugated = c.conj()

    assert type(conjugated) is CompositeQArray
    assert np.allclose(conjugated.to_jax(), oracle.conj(), atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_trace_stays_lazy(monkeypatch):
    c, oracle = _general_composite()
    _forbid_materialize(monkeypatch, '`trace` must not materialize')

    trace = c.trace()

    assert np.allclose(trace, np.trace(oracle, axis1=-2, axis2=-1), atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_mul_by_batched_scalar_stays_lazy(monkeypatch):
    c, oracle = _general_composite()
    y = jnp.array([1.0, 2.0j, -1.5])[:, None, None]
    _forbid_materialize(monkeypatch, '`__mul__` must not materialize')

    product = c * y

    assert type(product) is CompositeQArray
    monkeypatch.undo()
    expected = np.asarray(y) * oracle
    assert product.shape == expected.shape
    assert np.allclose(product.to_jax(), expected, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_mul_rejects_non_scalar():
    c, oracle = _general_composite()

    with pytest.raises(NotImplementedError, match='element-wise multiplication, use'):
        c * jnp.ones(oracle.shape[-2:])


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('build', 'stays_lazy'),
    [
        pytest.param(_single_term_composite, True, id='single-term'),
        pytest.param(_general_composite, False, id='multi-term'),
    ],
)
def test_powm(build, stays_lazy):
    c, oracle = build()
    n = 3

    powered = c.powm(n)

    assert isinstance(powered, CompositeQArray) is stays_lazy
    expected = np.linalg.matrix_power(oracle, n)
    assert np.allclose(powered.to_jax(), expected, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('build', 'stays_lazy'),
    [
        pytest.param(_single_term_composite, True, id='single-term'),
        pytest.param(_general_composite, False, id='multi-term'),
    ],
)
def test_elpow(build, stays_lazy):
    c, oracle = build()
    power = 3

    powered = c.elpow(power)

    assert isinstance(powered, CompositeQArray) is stays_lazy
    assert np.allclose(powered.to_jax(), oracle**power, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_expm_materializes():
    c, oracle = _general_composite()

    exponential = c.expm()

    assert type(exponential) is MaterializedQArray
    expected = jax.scipy.linalg.expm(jnp.asarray(oracle))
    assert np.allclose(exponential.to_jax(), expected, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_eig_single_term_stays_lazy(monkeypatch):
    c, oracle = _single_term_composite()
    _forbid_materialize(monkeypatch, '`_eig` must not materialize a single term')

    eigvals, V = c._eig()

    # H V = V diag(eigvals), column by column
    V = np.asarray(V.to_jax())
    assert np.allclose(oracle @ V, V * eigvals[..., None, :], atol=1e-5)
    assert np.allclose(c._eigvals(), eigvals)


@pytest.mark.run(order=TEST_SHORT)
def test_eigvalsh_single_term_stays_lazy(monkeypatch):
    c, oracle = _hermitian_term_composite()
    _forbid_materialize(monkeypatch, '`_eigvalsh` must not materialize a single term')

    eigvals = c._eigvalsh()

    assert np.allclose(eigvals, np.linalg.eigvalsh(oracle), atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_eigh_returns_orthonormal_eigenvectors():
    c, oracle = _hermitian_term_composite()
    n = oracle.shape[-1]

    eigvals, V = c._eigh()

    assert np.allclose(eigvals, np.linalg.eigvalsh(oracle), atol=1e-5)
    assert np.allclose(V.conj().T @ V, np.eye(n), atol=1e-5)
    assert np.allclose(V @ np.diag(eigvals) @ V.conj().T, oracle, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_norm():
    c, oracle = _hermitian_term_composite()

    # indefinite operator: the trace that `psd=True` returns is not the trace norm
    assert np.allclose(c.norm(psd=True), np.trace(oracle).real, atol=1e-5)
    expected = np.abs(np.linalg.eigvalsh(oracle)).sum()
    assert np.allclose(c.norm(psd=False), expected, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('build', 'expected'),
    [
        pytest.param(lambda: _batched_composite(dq.dense), True, id='factors'),
        pytest.param(_hermitian_term_composite, True, id='term'),
        pytest.param(_hermitian_sum_composite, True, id='cancellation'),
        pytest.param(_nearly_hermitian_composite, False, id='nearly-hermitian'),
        pytest.param(_general_composite, False, id='not-hermitian'),
    ],
)
def test_isherm(build, expected):
    c, oracle = build()

    oracle_dag = np.swapaxes(oracle.conj(), -1, -2)
    assert np.allclose(oracle, oracle_dag, rtol=1e-5, atol=1e-8) is expected
    assert bool(c.isherm(rtol=1e-5, atol=1e-8)) is expected


@pytest.mark.run(order=TEST_SHORT)
def test_isherm_with_hermitian_factors_stays_lazy(monkeypatch):
    c, _ = _batched_composite(dq.dense)
    _forbid_materialize(monkeypatch, '`isherm` must not materialize')

    assert bool(c.isherm())


@pytest.mark.run(order=TEST_SHORT)
def test_isherm_under_jit():
    # the sufficient check gives a tracer, which cannot decide a branch
    c, _ = _batched_composite(dq.dense)

    assert bool(jax.jit(lambda x: x.isherm())(c))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('layout', [dq.dense, dq.dia])
@pytest.mark.parametrize(
    ('keep', 'stays_lazy'),
    [
        pytest.param((0, 1), True, id='whole-operators'),
        pytest.param((2, 0), True, id='part-of-an-operator'),
        pytest.param((1,), False, id='single-subsystem'),
    ],
)
def test_ptrace(layout, keep, stays_lazy):
    c, oracle = _ptrace_composite(layout)

    reduced = c.ptrace(*keep)

    assert isinstance(reduced, CompositeQArray) is stays_lazy
    # like the partial trace of a materialized qarray, whatever the layout
    assert reduced.layout is dq.dense
    expected = dq.ptrace(oracle, keep, dims=c.dims)
    assert reduced.dims == expected.dims
    assert np.allclose(reduced.to_jax(), expected.to_jax(), atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    ('keep', 'match'),
    [
        pytest.param((), 'at least one subsystem', id='empty'),
        pytest.param((0, 0), 'duplicates', id='duplicate'),
        pytest.param((0, 3), 'Hilbert space structure', id='out-of-range'),
    ],
)
def test_ptrace_rejects_invalid_keep(keep, match):
    c, _ = _ptrace_composite(dq.dense)

    with pytest.raises(ValueError, match=match):
        c.ptrace(*keep)


@pytest.mark.run(order=TEST_SHORT)
def test_addscalar_batched_scalar_stays_lazy(monkeypatch):
    c, oracle = _general_composite()
    y = jnp.array([1.0, 2.0j, -1.5])[:, None, None]
    _forbid_materialize(monkeypatch, '`addscalar` must not materialize')

    shifted = c.addscalar(y)

    assert type(shifted) is CompositeQArray
    monkeypatch.undo()
    expected = oracle + np.asarray(y)
    assert shifted.shape == expected.shape
    assert np.allclose(shifted.to_jax(), expected, atol=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_addscalar_dia_warns_and_converts_to_dense():
    c, oracle = _batched_composite(dq.dia)
    y = 2.0

    with pytest.warns(UserWarning, match='converted to dense layout'):
        shifted = c.addscalar(y)

    assert type(shifted) is CompositeQArray
    assert shifted.layout is dq.dense
    assert np.allclose(shifted.to_jax(), oracle + y)


@pytest.mark.run(order=TEST_SHORT)
def test_addscalar_non_scalar_materializes():
    c, oracle = _general_composite()
    y = np.arange(oracle[0].size).reshape(oracle.shape[-2:])

    shifted = c.addscalar(y)

    assert type(shifted) is MaterializedQArray
    assert np.allclose(shifted.to_jax(), oracle + y, atol=1e-5)
