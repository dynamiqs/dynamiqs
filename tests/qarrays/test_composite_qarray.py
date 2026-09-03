import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.qarrays.composite_qarray import CompositeQArray, CompositeTerm
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
            lambda: CompositeTerm(operators=()),
            ValueError,
            id='term-operators-empty',
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
            lambda: CompositeQArray((2, 3), ()),
            ValueError,
            id='qarray-terms-empty',
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
        operators=(dq.asqarray(A0, dims=(2,)), dq.asqarray(B0, dims=(3,))),
        coeff=coeff0,
    )
    term1 = CompositeTerm(
        operators=(dq.asqarray(A1, dims=(2,)), dq.asqarray(B1, dims=(3,))),
        coeff=coeff1,
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
