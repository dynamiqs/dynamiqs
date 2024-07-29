from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from .dense_qarray import DenseQArray
from .sparse_dia_qarray import SparseDIAQArray
from .types import QArray

__all__ = ['stack']


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
        >>> dq.stack([dq.fock(3, 0), dq.fock(3, 1)]).shape
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
        offsets = qarrays[0].offsets
        if not all(q.offsets == offsets for q in qarrays):
            # TODO: implement stacking with different offsets
            raise ValueError(
                'Argument `qarrays` with elements of type `SparseDIAQArray` must have'
                ' identical `offsets` attribute.'
            )
        diags = jnp.stack([q.diags for q in qarrays], axis=axis)
        return SparseDIAQArray(dims, offsets, diags)
    else:
        raise NotImplementedError(
            'Stacking qarrays with different types is not implemented.'
        )
