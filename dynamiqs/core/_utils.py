from __future__ import annotations

from typing import get_args

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..solver import Solver
from ..time_array import TimeArray, _factory_constant
from .abstract_solver import AbstractSolver


def _astimearray(
    x: ArrayLike | TimeArray, dtype: jnp.complex64 | jnp.complex128
) -> TimeArray:
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype)
    elif isinstance(x, TimeArray):
        return x
    else:
        raise TypeError()  # todo: add error message


def get_solver_class(
    solvers: dict[Solver, AbstractSolver], solver: Solver
) -> AbstractSolver:
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    return solvers[type(solver)]


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
            # we apply the successive vmaps in reverse order, so that the output
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
