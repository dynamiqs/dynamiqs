from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from qutip import Qobj

from .._checks import check_shape
from .._utils import cdtype
from ..utils.quantum_utils import isbra, isket, isop
from ..utils.quantum_utils.general import _hdim
from .dense_qarray import DenseQArray
from .sparse_dia_qarray import SparseDIAQArray
from .types import QArray, QArrayLike

__all__ = ['stack', 'to_qutip']


def stack(qarrays: Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of qarrays along a new axis.

    Warning:
        All elements of the sequence `qarrays` must have identical types, shapes and
        `dims` attributes. Additionally, when stacking qarrays of type
        `SparseDIAQArray`, all elements must have identical `offsets` attributes.

    Args:
        qarrays: Qarrays to stack.
        axis: Axis in the result along which the input qarrays are stacked.

    Returns:
        Stacked qarray.

    Examples:
        >>> dq.stack([dq.fock(3, 0), dq.fock(3, 1)])
        DenseQArray: shape=(2, 3, 1), dims=(3,), dtype=complex64
        [[[1.+0.j]
          [0.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [1.+0.j]
          [0.+0.j]]]
    """
    # check validity of input
    if len(qarrays) == 0:
        raise ValueError('Argument `qarrays` must contain at least one element.')
    if not all(isinstance(q, QArray) for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain only elements of type `QArray`.'
        )
    dims = qarrays[0].dims
    if not all(q.dims == dims for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `dims` attribute.'
        )

    # stack inputs depending on type
    if all(isinstance(q, DenseQArray) for q in qarrays):
        data = jnp.stack([q.data for q in qarrays], axis=axis)
        return DenseQArray(dims, data)
    elif all(isinstance(q, SparseDIAQArray) for q in qarrays):
        unique_offsets = set()
        for qarray in qarrays:
            unique_offsets.update(qarray.offsets)
        unique_offsets = tuple(sorted(unique_offsets))

        offset_to_index = {offset: idx for idx, offset in enumerate(unique_offsets)}
        diag_list = []
        for qarray in qarrays:
            add_diags_shape = qarray.diags.shape[:-2] + (
                len(unique_offsets),
                qarray.diags.shape[-1],
            )
            updated_diags = jnp.zeros(add_diags_shape, dtype=cdtype())
            for i, offset in enumerate(qarray.offsets):
                idx = offset_to_index[offset]
                updated_diags = updated_diags.at[..., idx, :].set(
                    qarray.diags[..., i, :]
                )
            diag_list.append(updated_diags)
        return SparseDIAQArray(dims, unique_offsets, jnp.stack(diag_list))
    else:
        raise NotImplementedError(
            'Stacking qarrays with different types is not implemented.'
        )


def to_qutip(x: QArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert an array-like object into a QuTiP quantum object (or a list of QuTiP
    quantum objects if it has more than two dimensions).

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the large
            Hilbert space of the composite system, defaults to `None` (a single system
            with the same dimension as `x`).

    Returns:
        QuTiP quantum object or list of QuTiP quantum objects.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        DenseQArray: shape=(3, 1), dims=(3,), dtype=complex64
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]
        >>> dq.to_qutip(psi)
        Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
        Qobj data =
        [[0.]
         [1.]
         [0.]]
        For a batched array:
        >>> rhos = dq.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)
        >>> len(dq.to_qutip(rhos))
        5
        Note that the tensor product structure is inferred automatically for qarrays. It
        can be specified with the `dims` argument for other types.
        >>> I = dq.eye(3, 2)
        >>> dq.to_qutip(I)
        Quantum object: dims = [[3, 2], [3, 2]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    if isinstance(x, QArray):
        dims = x.dims

    x = np.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if x.ndim > 2:
        return [to_qutip(sub_x, dims=dims) for sub_x in x]
    else:
        dims = [_hdim(x)] if dims is None else list(dims)
        if isket(x):  # [[3], [1]] or for composite systems [[3, 4], [1, 1]]
            dims = [dims, [1] * len(dims)]
        elif isbra(x):  # [[1], [3]] or for composite systems [[1, 1], [3, 4]]
            dims = [[1] * len(dims), dims]
        elif isop(x):  # [[3], [3]] or for composite systems [[3, 4], [3, 4]]
            dims = [dims, dims]
        return Qobj(x, dims=dims)
