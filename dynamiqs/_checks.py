from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_is_perfect_square = lambda n: int(n**0.5) ** 2 == n


def has_shape(x: Array, shape: str) -> bool:
    cases = {
        '(..., n, 1)': lambda x: x.ndim >= 2 and x.shape[-1] == 1,
        '(..., 1, n)': lambda x: x.ndim >= 2 and x.shape[-2] == 1,
        '(..., n, n)': lambda x: x.ndim >= 2 and x.shape[-2] == x.shape[-1],
        '(N, ..., n, n)': lambda x: x.ndim >= 3 and x.shape[-2] == x.shape[-1],
        '(..., n)': lambda x: x.ndim >= 1,
        '(n, 1)': lambda x: x.ndim == 2 and x.shape[-1] == 1,
        '(1, n)': lambda x: x.ndim == 2 and x.shape[-2] == 1,
        '(n, n)': lambda x: x.ndim == 2 and x.shape[-2] == x.shape[-1],
        '(n,)': lambda x: x.ndim == 1,
        '(N, n, 1)': lambda x: x.ndim == 3 and x.shape[-1] == 1,
        '(N, n, n)': lambda x: x.ndim == 3 and x.shape[-2] == x.shape[-1],
        '(?, n, n)': lambda x: 2 <= x.ndim <= 3 and x.shape[-2] == x.shape[-1],
        '(..., n^2, 1)': lambda x: x.ndim >= 2
        and _is_perfect_square(x.shape[-2])
        and x.shape[-1] == 1,
    }
    if shape in cases:
        return cases[shape](x)
    else:
        raise ValueError(f'Unknown shape specification `{shape}`.')


def check_shape(
    x: Array, argname: str, *shapes: str, subs: dict[str, str] | None = None
):
    for shape in shapes:
        if has_shape(x, shape):
            return

    if len(shapes) == 1:
        shapes_str = shapes[0]
    else:
        shapes_str = ', '.join(shapes[:-1]) + ' or ' + shapes[-1]

    if subs is not None:
        for k, v in subs.items():
            shapes_str = shapes_str.replace(k, v)

    raise ValueError(
        f'Argument `{argname}` must have shape {shapes_str}, but has shape'
        f' {argname}.shape={x.shape}.'
    )


def check_time_array(x: Array, argname: str, allow_empty: bool = False):
    # check that a time-array is valid (it must be a 1D array sorted in strictly
    # ascending order and containing only positive values)
    # check_shape(x, argname, '(n,)', subs={'n': f'n{argname}'})

    if x.ndim != 1:
        raise ValueError(
            f'Argument {argname} must be a 1D array, but is a {x.ndim}D array.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument {argname} must contain at least one element.')
    if not jnp.all(jnp.diff(x) > 0):
        raise ValueError(
            f'Argument {argname} must be sorted in strictly ascending order.'
        )
    if not jnp.all(x >= 0):
        raise ValueError(f'Argument {argname} must contain positive values only.')
