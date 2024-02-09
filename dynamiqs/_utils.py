from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array

from .utils.utils import dag, isket


def type_str(type: Any) -> str:
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def check_time_array(x: Array, arg_name: str, allow_empty: bool = False):
    # check that a time array is valid (it must be a 1D array sorted in strictly
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


def bexpect(O: Array, x: Array) -> Array:
    # batched over O
    if isket(x):
        return jnp.einsum('ij,bjk,kl->b', dag(x), O, x)  # <x|O|x>
    return jnp.einsum('bij,ji->b', O, x)  # tr(Ox)


def compute_vmap(
    f: callable,
    cartesian_batching: bool,
    is_batched: list[bool],
    out_axes: list[int | None],
) -> callable:
    if any(is_batched):
        if cartesian_batching:
            # iteratively map over the first axis of each batched argument
            idx_batched = np.where(is_batched)[0]
             # we apply the succesive vmaps in reverse order, so that the output
             # batched dimensions are in the correct order
            for i in reversed(idx_batched):
                in_axes = [None] * len(is_batched)
                in_axes[i] = 0
                f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
        else:
            # map over the first axis of all batched arguments
            in_axes = list(np.where(is_batched, 0, None))
            f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f
