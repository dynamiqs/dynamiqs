from __future__ import annotations

from typing import Any, Union, get_args

import jax
import jax.numpy as jnp
import jaxtyping
from qutip import Qobj

from .qarrays.qarray import QArray

# In this file we define two extended array types: `ArrayLike` and `Array`. Most
# functions in the library take an `ArrayLike` as argument and return an `Array`.
# `ArrayLike` can be any numeric type (bool, int, float, complex), a JAX Array, a NumPy
# Array, a QuTiP Qobj, a dynamiqs QArray or a nested list of these types. Any
# `ArrayLike` can be converted to an `Array` using `asarray`, resulting in either a JAX
# Array or a dynamiqs QArray.

# extended array like type
_ArrayLike = Union[jaxtyping.ArrayLike, QArray, Qobj]
# a type alias for nested list of _ArrayLike
_NestedArrayLikeList = list[Union[_ArrayLike, '_NestedArrayLikeList']]
# a type alias for any type compatible with asarray
ArrayLike = Union[_ArrayLike, _NestedArrayLikeList]

Array = Union[jax.Array, QArray]


def is_arraylike(x: Any) -> bool:
    if isinstance(x, get_args(_ArrayLike)):
        return True
    elif isinstance(x, list):
        return all(isinstance(_x, get_args(_ArrayLike)) for _x in x)
    return False


def asarray(x: ArrayLike) -> Array:
    return x if isinstance(x, QArray) else jnp.asarray(x)


def asjaxarray(x: ArrayLike) -> jax.Array:
    return x.to_jax() if isinstance(x, QArray) else jnp.asarray(x)
