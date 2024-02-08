from __future__ import annotations

from typing import Any

import jax
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


def compute_batching(f, cartesian_batching, other_in_axes, out_axes, *should_batch):
    # Build a list of batch mask (ie binary numbers). Each digit is 1 if we
    # should batch over the dimension and 0 otherwise.
    # For example if `should_batch` = (True, False, True), the batch_mask will be
    # [001, 100]
    batch_mask = [2**k * v for k, v in enumerate(should_batch)]

    # When doing cartesian batching, we need to batch over all dimensions
    # one by one, but when cartesian batching is off, we want to batch over
    # every dimension at once, hence the sum.
    # For example if `should_batch` = (True, False, True), the batch_mask will be
    # [001, 100] if we want to do cartesian batching and [101] otherwise.
    if not cartesian_batching:
        batch_mask = [sum(batch_mask)]

    for mask in reversed(batch_mask):
        if mask == 0:  # skip the no batching case
            continue
        mask = tuple([0 if mask & (1 << i) else None for i in range(len(should_batch))])
        f = jax.vmap(f, in_axes=mask + other_in_axes, out_axes=out_axes)

    return f
