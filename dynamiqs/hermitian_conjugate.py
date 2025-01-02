from __future__ import annotations

from ._utils import obj_type_str
from .qarrays import QArray
from .time_qarray import TimeQArray

__all__ = ['hc']


class HermitianConjugate:
    def __add__(self, _):  # noqa
        raise TypeError(
            'The symbol `dq.hq` can only be right-added to a qarray or a time-qarray.'
        )

    def __radd__(self, y: QArray | TimeQArray) -> QArray | TimeQArray:
        if isinstance(y, QArray | TimeQArray):
            return y + y.dag()
        else:
            raise TypeError(
                f'The symbol `dq.hq` can only be right-added to a qarray or a '
                f'time-qarray, but it was added to a {obj_type_str(y)}.'
            )


hc = HermitianConjugate()
"""This symbol can be used as a shortcut to sum a qarray or a time-qarray with its
Hermitian conjugate.

Examples:
    For a linear drive with amplitude `omega`:
    ```diff
    - H = a.dag() @ a + omega * a + omega.conj() * a.dag()
    + H = a.dag() @ a + (omega * a + dq.hc)
    ```

    For two linear drives on two different modes:
    ```diff
    - H = omega_a * a + omega_a.conj() * a.dag() + omega_b * b + omega_b.conj() * b
    + H = (omega_a * a + dq.hc) + (omega_b * b + dq.hc)
    ```

    For a time-dependent linear drive:
    ```diff
    omega = lambda t: jnp.cos(2.0 * jnp.pi * t)
    - H = dq.modulated(omega, a)
    - H += H.dag()
    + H = dq.modulated(omega, a) + dq.hc
    ```

Warning:
    The only valid operation with this symbol is right-addition to a qarray or a
    time-qarray. Any other operation will raise a `TypeError`.

Warning:
    The symbol applies on all terms present in the left-hand side of the addition. If
    you want to apply it to a single term, you should use parentheses to isolate it:

    ```pycon
    >>> sx, sz = dq.sigmax(), dq.sigmaz()
    >>> sz + (sx + dq.hc)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=3
    [[ 1.+0.j  2.+0.j]
     [ 2.+0.j -1.+0.j]]
    >>> sz + sx + dq.hc
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=3
    [[ 2.+0.j  2.+0.j]
     [ 2.+0.j -2.+0.j]]
    ```
"""
