from __future__ import annotations

from typing import Any, Union, get_args

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, DTypeLike
from qutip import Qobj

from .qarray import QArray

__all__ = [
    'QArray',
    'QArrayLike',
    'asqarray',
    'asjaxarray',
    'isqarraylike',
    'sparsedia',
]

# In this file we define an extended array type named `QArrayLike`. Most
# functions in the library take a `QArrayLike` as argument and return a `QArray`.
# `QArrayLike` can be:
# - any numeric type (bool, int, float, complex),
# - a JAX array,
# - a NumPy array,
# - a QuTiP Qobj,
# - a dynamiqs QArray,
# - a nested list of these types.
# An object of type `QArrayLike` can be converted to a `QArray` with `asqarray`.

# extended array like type
_QArrayLike = Union[ArrayLike, QArray, Qobj]
# a type alias for nested list of _QArrayLike
_NestedQArrayLikeList = list[Union[_QArrayLike, '_NestedQArrayLikeList']]
# a type alias for any type compatible with asqarray
QArrayLike = Union[_QArrayLike, _NestedQArrayLikeList]


def isqarraylike(x: Any) -> bool:
    if isinstance(x, get_args(_QArrayLike)):
        return True
    elif isinstance(x, list):
        return all(isqarraylike(_x) for _x in x)
    return False


def asqarray(x: QArrayLike, dims: int | tuple[int, ...] | None = None) -> QArray:
    if isinstance(x, QArray):
        return x

    # TODO: improve this fix
    if isinstance(x, list) and all(isinstance(x_, QArray) for x_ in x):
        from .utils import stack

        return stack(x)

    from .dense_qarray import DenseQArray

    # TODO: check if is bra, ket, dm or op
    # if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
    # raise ValueError(
    #     f'DenseQArray data must be a bra, a ket, a density matrix '
    #     f'or and operator. Got array with size {data.shape}'
    # )
    # TODO: check that dims and shape work

    x = jnp.asarray(x)
    if dims is None:
        dims = (x.shape[-2],) if x.shape[-2] != 1 else (x.shape[-1],)
    elif isinstance(dims, int):
        dims = (dims,)
    return DenseQArray(dims, x)


def asjaxarray(x: QArrayLike) -> Array:
    return x.to_jax() if isinstance(x, QArray) else jnp.asarray(x)


def sparsedia(
    offsets_diags: dict[int, ArrayLike],
    dims: int | tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> QArray:  # todo: fix return type (circular import)
    from .sparse_dia_qarray import SparseDIAQArray

    # === offsets
    offsets = tuple(offsets_diags.keys())

    # === diags
    # stack arrays in a square matrix by padding each according to its offset
    pads_width = [(abs(k), 0) if k >= 0 else (0, abs(k)) for k in offsets]
    diags = [jnp.asarray(diag) for diag in offsets_diags.values()]
    diags = [jnp.pad(diag, pad_width) for pad_width, diag in zip(pads_width, diags)]
    diags = jnp.stack(diags, dtype=dtype)

    # === dims
    if dims is None:
        dims = (diags.shape[-1],)
    elif isinstance(dims, int):
        dims = (dims,)

    return SparseDIAQArray(diags=diags, offsets=offsets, dims=dims)
