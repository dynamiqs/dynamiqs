import jax.numpy as jnp
import pytest
import qutip as qt

import dynamiqs as dq

from ..order import TEST_INSTANT

pytestmark = [
    pytest.mark.run(order=TEST_INSTANT),
    pytest.mark.parametrize('layout', [dq.dense, dq.dia]),
]

OPS = [
    ('add', lambda x, y: x + y),
    ('sub', lambda x, y: x - y),
    ('mul', lambda x, y: x * y),
    ('truediv', lambda x, y: x / y),
    ('matmul', lambda x, y: x @ y),
    ('pow', lambda x, y: x**y),
]


class _ReflectedOnly:
    """Object that only implements reflected arithmetic dunders.

    Used to check that qarray arithmetic operators correctly return
    `NotImplemented` for operands they don't understand, so that Python falls
    back to the other operand's reflected method instead of raising.
    """

    def __radd__(self, other):
        return ('add', other)

    def __rsub__(self, other):
        return ('sub', other)

    def __rmul__(self, other):
        return ('mul', other)

    def __rtruediv__(self, other):
        return ('truediv', other)

    def __rmatmul__(self, other):
        return ('matmul', other)

    def __rpow__(self, other):
        return ('pow', other)


@pytest.fixture
def x(layout):
    return dq.asqarray(dq.sigmax(), layout=layout)


@pytest.mark.parametrize(('name', 'op'), OPS, ids=[name for name, _ in OPS])
def test_arithmetic_falls_back_to_reflected_method(x, name, op):
    assert op(x, _ReflectedOnly()) == (name, x)


@pytest.mark.parametrize('op', [op for _, op in OPS], ids=[name for name, _ in OPS])
def test_arithmetic_with_unsupported_type_raises_typeerror(x, op):
    # objects with no arithmetic support at all should raise a plain `TypeError`
    # (Python's usual error for unsupported operand types), not crash or hang
    with pytest.raises(TypeError):
        op(x, object())


@pytest.mark.parametrize(
    'y',
    [dq.sigmay(), qt.sigmax(), [[1, 2], [3, 4]], jnp.array([[1.0, 2.0], [3.0, 4.0]])],
    ids=['qarray', 'qobj', 'nested_list', 'raw_array'],
)
def test_truediv_by_non_scalar_raises_cleanly(x, y):
    # dividing a qarray by a non-scalar qarray-like must raise a clear
    # `NotImplementedError`, not crash inside the `1 / y` reciprocal computation
    # (e.g. `1 / qutip.Qobj` used to raise a confusing raw `TypeError`)
    with pytest.raises(NotImplementedError):
        x / y


def test_arithmetic_between_two_qarrays_still_raises(x, layout):
    # legitimate error messages for unsupported qarray-qarray operations must
    # still raise `NotImplementedError`, not silently fall back
    y = dq.asqarray(dq.sigmay(), layout=layout)
    with pytest.raises(NotImplementedError):
        x * y
    with pytest.raises(NotImplementedError):
        1 / x
    with pytest.raises(NotImplementedError):
        x**2
