from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import jax.numpy as jnp

from .._utils import cdtype
from .dense_qarray import DenseQArray
from .sparse_dia_qarray import SparseDIAQArray
from .types import QArray

__all__ = ['stack', 'set_layout']


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


class Layout(Enum):
    DENSE = 0
    DIA = 1


dense = Layout.DENSE
dia = Layout.DIA

_DEFAULT_LAYOUT = dia


def set_layout(layout: Layout):
    """Configure the default matrix layout for operators supporting this option.

    Two layouts are supported by most operators (see the list of available operators in
    the [Python API](../../index.md#operators)):

    - `dq.dense`: JAX native dense layout,
    - `dq.dia`: dynamiqs sparse diagonal layout, only non-zero diagonals are stored.

    Note:
        The default layout upon importing dynamiqs is `dq.dia`.

    Args:
        layout: Default matrix layout for operators (`dq.dense` or `dq.dia`).

    Examples:
        >>> dq.eye(4)
        SparseDIAQArray: shape=(4, 4), dims=(4,), dtype=complex64, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]
        >>> dq.set_layout(dq.dense)
        >>> dq.eye(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """
    if not isinstance(layout, Layout):
        raise TypeError(
            f'Argument `layout` must be `dq.dense` or `dq.dia`, but is `{layout}`'
        )

    global _DEFAULT_LAYOUT  # noqa: PLW0603
    _DEFAULT_LAYOUT = layout


def get_layout(layout: Layout | None = None) -> Layout:
    if layout is None:
        return _DEFAULT_LAYOUT
    elif isinstance(layout, Layout):
        return layout
    else:
        raise TypeError(
            'Argument `layout` must be `dq.dense`, `dq.dia` or `None`, but is'
            f' `{layout}`.'
        )