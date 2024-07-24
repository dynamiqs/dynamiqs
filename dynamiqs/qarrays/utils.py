from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from .dense_qarray import DenseQArray
from .sparse_dia_qarray import SparseDIAQArray
from .types import QArray

__all__ = ['stack']


def stack(qarrays: QArray | Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of QArrays along a new axis.

    Args:
        qarrays: The QArrays to be stacked.
        axis: The axis in the result along which the input QArrays are stacked.

    Returns:
        QArray: The stacked QArray.
    """
    if isinstance(qarrays, QArray):
        return qarrays

    # check validity of input
    if len(qarrays) == 0:
        raise ValueError('At least one QArray must be provided.')
    if not all(isinstance(q, QArray) for q in qarrays):
        raise ValueError('All elements must be of type QArray.')
    if not all(qarray.dims == qarrays[0].dims for qarray in qarrays):
        raise ValueError('All elements must have the same dims.')

    # stack inputs depending on type
    dims = qarrays[0].dims
    if all(isinstance(qarray, DenseQArray) for qarray in qarrays):
        data = jnp.stack([qarray.data for qarray in qarrays], axis=axis)
        return DenseQArray(dims, data)
    elif all(isinstance(qarray, SparseDIAQArray) for qarray in qarrays):
        if not all(qarray.offsets == qarrays[0].offsets for qarray in qarrays):
            # todo: implement stacking with different offsets
            raise NotImplementedError(
                'All SparseDIAQArrays to be stacked must have the same offsets.'
            )
        diags = jnp.stack([x.diags for x in qarrays], axis=axis)
        offsets = qarrays[0].offsets
        return SparseDIAQArray(dims, diags, offsets)
    else:
        raise NotImplementedError(
            'Stacking different types of QArrays is not implemented.'
        )
