from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


def type_str(type: Any) -> str:  # noqa: A002
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def check_time_array(x: Array, arg_name: str, allow_empty: bool = False):
    # check that a time-array is valid (it must be a 1D array sorted in strictly
    # ascending order and containing only positive values)
    if x.ndim != 1:
        raise ValueError(
            f'Argument `{arg_name}` must be a 1D array, but is a {x.ndim}D array.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument `{arg_name}` must contain at least one element.')
    if not jnp.all(jnp.diff(x) > 0):
        raise ValueError(
            f'Argument `{arg_name}` must be sorted in strictly ascending order.'
        )
    if not jnp.all(x >= 0):
        raise ValueError(f'Argument `{arg_name}` must contain positive values only.')


def on_cpu(x: Array) -> str:
    # TODO: this is a temporary solution, it won't work when we have multiple devices
    return x.devices().pop().device_kind == 'cpu'


def _get_default_dtype() -> jnp.float32 | jnp.float64:
    default_dtype = jnp.array(0.0).dtype
    return jnp.float64 if default_dtype == jnp.float64 else jnp.float32


def cdtype() -> jnp.complex64 | jnp.complex128:
    # the default dtype for complex arrays is determined by the default floating point
    # dtype
    dtype = _get_default_dtype()
    if dtype is jnp.float32:
        return jnp.complex64
    elif dtype is jnp.float64:
        return jnp.complex128
    else:
        raise ValueError(f'Data type `{dtype.dtype}` is not yet supported.')
