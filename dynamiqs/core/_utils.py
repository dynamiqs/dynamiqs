from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._utils import cdtype, obj_type_str
from ..solver import Solver
from ..time_array import ConstantTimeArray, TimeArray
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
                f'Argument must be an array-like or a time-array object, but has type'
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
    f: callable,
    cartesian_batching: bool,
    is_batched: list[bool | list[bool]],
    out_axes: list[int | None],
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

    leaves, treedef = jax.tree_util.tree_flatten(is_batched)
    n = len(leaves)
    if any(leaves):
        if cartesian_batching:
            # map over each batched dimension separately
            # note: we apply the successive vmaps in reverse order, so the output
            # batched dimensions are in the correct order
            for i, leaf in enumerate(reversed(leaves)):
                if leaf:
                    # build the `in_axes` argument with the same structure as
                    # `is_batched`, but with 0 at the `leaf` position
                    in_axes = jax.tree_util.tree_map(lambda _: None, leaves)
                    in_axes[n - 1 - i] = 0
                    in_axes = jax.tree_util.tree_unflatten(treedef, in_axes)
                    f = jax.vmap(f, in_axes=in_axes, out_axes=0)
        else:
            # map over all batched dimensions at once
            in_axes = jax.tree_util.tree_map(lambda x: 0 if x else None, is_batched)
            f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f
