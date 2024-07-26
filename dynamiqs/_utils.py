from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree


def type_str(type: Any) -> str:  # noqa: A002
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def on_cpu(x: Array) -> bool:
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


def tree_str_inline(x: PyTree) -> str:
    # return an inline formatting of a pytree as a string
    return eqx.tree_pformat(x, indent=0).replace('\n', '').replace(',', ', ')


def _concatenate_sort(*args: Array | None) -> Array | None:
    args = [x for x in args if x is not None]
    if len(args) == 0:
        return None
    return jnp.sort(jnp.concatenate(args))
