from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree

from .._utils import cdtype, obj_type_str
from ..solver import Solver
from ..time_array import (
    CallableTimeArray,
    ConstantTimeArray,
    PWCTimeArray,
    SummedTimeArray,
    TimeArray,
)
from .abstract_solver import AbstractSolver


def _astimearray(x: ArrayLike | TimeArray) -> TimeArray:
    if isinstance(x, TimeArray):
        return x
    else:
        try:
            # same as dq.constant() but not checking the shape
            array = jnp.asarray(x, dtype=cdtype())
            return ConstantTimeArray(array)
        except (TypeError, ValueError) as e:
            raise TypeError(
                'Argument must be an array-like or a time-array object, but has type'
                f' {obj_type_str(x)}.'
            ) from e


def get_solver_class(
    solvers: dict[Solver, AbstractSolver], solver: Solver
) -> AbstractSolver:
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers)
        raise TypeError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    return solvers[type(solver)]


def compute_vmap(
    f: callable, cartesian_batching: bool, n_batch: [int], out_axes: PyTree[int | None]
) -> callable:
    # This function vectorizes `f` by applying jax.vmap over batched dimensions. The
    # argument `is_batched` indicates for each argument of `f` whether it is batched.
    # There are two possible strategies to vectorize `f`:
    # - If `cartesian_batching` is True, we want to apply `f` to every possible
    #   combination of the arguments batched dimension (the cartesian product). To do
    #   so, we essentially wrap f with multiple vmap applications, one for each batched
    #   dimension.
    # - If `cartesian_batching` is False, we directly map f over all batched arguments
    #   and apply vmap once.

    # we use tree_utils to handle nested batching such as `jump_ops`
    leaves, treedef = jtu.tree_flatten(n_batch)
    if any(leaf > 0 for leaf in leaves):
        if cartesian_batching:
            # map over each batched dimension separately
            # note: we apply the successive vmaps in reverse order, so the output
            # batched dimensions are in the correct order
            for i, leaf in reversed(list(enumerate(leaves))):
                if leaf > 0:
                    # build the `in_axes` argument with the same structure as
                    # `is_batched`, but with 0 at the `leaf` position
                    in_axes = jtu.tree_map(lambda _: None, leaves)
                    in_axes[i] = 0
                    in_axes = jtu.tree_unflatten(treedef, in_axes)
                    for _ in range(leaf):
                        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
        else:
            # map over all batched dimensions at once
            in_axes = jtu.tree_map(lambda x: 0 if x > 0 else None, n_batch)
            n = jtu.tree_reduce(max, n_batch)
            for _ in range(n):
                f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f


def is_timearray_batched(tarray: TimeArray) -> TimeArray:
    # This function finds all batched arrays within a given TimeArray.
    # To do so, it goes down the PyTree and identifies batched fields depending
    # on the type of TimeArray.
    if isinstance(tarray, SummedTimeArray):
        return SummedTimeArray([is_timearray_batched(arr) for arr in tarray.timearrays])
    elif isinstance(tarray, ConstantTimeArray):
        return ConstantTimeArray(tarray.array.ndim > 2)
    elif isinstance(tarray, PWCTimeArray):
        return PWCTimeArray(False, tarray.values.ndim > 1, False)
    elif isinstance(tarray, CallableTimeArray):
        return CallableTimeArray(False, tuple(arg.ndim > 0 for arg in tarray.args))
    else:
        raise TypeError(f'Unsupported TimeArray type: {type(tarray).__name__}')
