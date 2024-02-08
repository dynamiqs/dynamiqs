from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from qutip import Qobj

from .utils import _hdim, isbra, isket, isop

__all__ = ['to_qutip']


def _get_default_dtype() -> jnp.float32 | jnp.float64:
    default_dtype = jnp.array(0.0).dtype
    return jnp.float64 if default_dtype == jnp.float64 else jnp.float32


def cdtype() -> jnp.complex64 | jnp.complex128:
    # the default dtype for complex arrays is determined by the default floating point
    # dtype (jnp.complex128 if default is jnp.float64, otherwise jnp.complex64)
    default_dtype = _get_default_dtype()
    return jnp.complex128 if default_dtype is jnp.float64 else jnp.complex64


def to_qutip(x: ArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert an array-like object into a QuTiP quantum object (or a list of QuTiP
    quantum objects if it has more than two dimensions).

    Args:
        x: Array-like object.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the large
            Hilbert space of the composite system, defaults to `None` (a single system
            with the same dimension as `x`).

    Returns:
        QuTiP quantum object or list of QuTiP quantum objects.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_qutip(psi)
        Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched array:
        >>> rhos = jnp.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)
        >>> len(dq.to_qutip(rhos))
        5

        Note that the tensor product structure is not inferred automatically, it must be
        specified with the `dims` argument:
        >>> I = dq.eye(3, 2)
        >>> dq.to_qutip(I)
        Quantum object: dims = [[6], [6]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
        >>> dq.to_qutip(I, (3, 2))
        Quantum object: dims = [[3, 2], [3, 2]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    x = np.asarray(x)

    if x.ndim > 2:
        return [to_qutip(sub_x) for sub_x in x]
    else:
        dims = [_hdim(x)] if dims is None else list(dims)
        if isket(x):  # [[3], [1]] or for composite systems [[3, 4], [1, 1]]
            dims = [dims, [1] * len(dims)]
        elif isbra(x):  # [[1], [3]] or for composite systems [[1, 1], [3, 4]]
            dims = [[1] * len(dims), dims]
        elif isop(x):  # [[3], [3]] or for composite systems [[3, 4], [3, 4]]
            dims = [dims, dims]
        return Qobj(x, dims=dims)
