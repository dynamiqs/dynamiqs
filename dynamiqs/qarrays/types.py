from __future__ import annotations

from typing import Any, Union, get_args

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike
from qutip import Qobj

from .qarray import QArray

__all__ = ['QArray', 'QArrayLike', 'asqarray', 'asjaxarray', 'dense_qarray']

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


def is_qarraylike(x: Any) -> bool:
    if isinstance(x, get_args(_QArrayLike)):
        return True
    elif isinstance(x, list):
        return all(is_qarraylike(_x) for _x in x)
    return False


def asqarray(x: QArrayLike, dims: int | tuple[int, ...] | None = None) -> QArray:
    return x if isinstance(x, QArray) else dense_qarray(x, dims=dims)


def dense_qarray(x: ArrayLike, dims: int | tuple[int, ...] | None = None) -> QArray:
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
        dims = (x.shape[-2],)  # TODO: fix for bra
    elif isinstance(dims, int):
        dims = (dims,)
    return DenseQArray(dims, x)


def asjaxarray(x: QArrayLike) -> Array:
    return x.to_jax() if isinstance(x, QArray) else jnp.asarray(x)
