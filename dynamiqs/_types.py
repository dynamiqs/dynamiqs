from __future__ import annotations

from typing import Any, Union, get_args

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from qutip import Qobj

from .qarrays.qarray import QArray

# In this file we define two extended array types: `QArrayLike` and `Array`. Most
# functions in the library take a `QArrayLike` as argument and return an `Array`.
# `QArrayLike` can be any numeric type (bool, int, float, complex), a JAX Array, a NumPy
# Array, a QuTiP Qobj, a dynamiqs QArray or a nested list of these types. Any
# `QArrayLike` can be converted to an `Array` using `asarray`, resulting in either a JAX
# Array or a dynamiqs QArray.

# extended array like type
_QArrayLike = Union[ArrayLike, QArray, Qobj]
# a type alias for nested list of _QArrayLike
_NestedQArrayLikeList = list[Union[_QArrayLike, '_NestedQArrayLikeList']]
# a type alias for any type compatible with asarray
QArrayLike = Union[_QArrayLike, _NestedQArrayLikeList]

Array = Union[jax.Array, QArray]


def is_qarraylike(x: Any) -> bool:
    if isinstance(x, get_args(_QArrayLike)):
        return True
    elif isinstance(x, list):
        return all(isinstance(_x, get_args(_QArrayLike)) for _x in x)
    return False


def asarray(x: QArrayLike) -> Array:
    return x if isinstance(x, QArray) else jnp.asarray(x)


def asjaxarray(x: QArrayLike) -> jax.Array:
    return x.to_jax() if isinstance(x, QArray) else jnp.asarray(x)
