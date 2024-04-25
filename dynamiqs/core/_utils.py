from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree

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


def _flat_vectorize(
    f: callable, n_batch: PyTree[int], out_axes: PyTree[int | None]
) -> callable:
    """Returns a vectorized function mapped over multiple axes (similarly to
    `jnp.vectorize`).

    The function is mapped on multiple axes, according to numpy broadcasting rules. This
    is achieved by nesting calls to `jax.vmap` for each leading dimensions specified by
    `n_batch`.

    Args:
        `n_batch`: PyTree indicating, for each argument of `f`, the number of leading
            dimensions that should be mapped over.
        `out_axes`: Equivalent of `out_axes` of `jax.vmap`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dynamiqs.core._utils import _flat_vectorize
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> n = 2
        >>> x = jnp.ones((3, 4, n, n))

        >>> y = jnp.ones((n, n))
        >>> f = _flat_vectorize(func, (2, 0), 0)
        >>> f(x, y).shape
        (3, 4, 2, 2)

        >>> y = jnp.ones((4, n, n))
        >>> f = _flat_vectorize(func, (2, 1), 0)
        >>> f(x, y).shape
        (3, 4, 2, 2)

        >>> y = jnp.ones((3, 1, n, n))
        >>> f = _flat_vectorize(func, (2, 2), 0)
        >>> f(x, y).shape
        (3, 4, 2, 2)
    """
    # todo: fix
    in_axes = jtu.tree_map(lambda x: 0 if x > 0 else None, n_batch)
    n = jtu.tree_reduce(max, n_batch)

    for _ in range(n):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f


def _cartesian_vectorize(
    f: callable, n_batch: PyTree[int], out_axes: PyTree[int | None]
) -> callable:
    """Returns a vectorized function mapped over all combinations of specified axes.

    The function is mapped on every combinations of axes (the cartesian product). This
    is achieved by nesting calls to `jax.vmap` for each argument and for each leading
    dimensions specified by `n_batch`.

    Args:
        `n_batch`: PyTree indicating, for each argument of `f`, the number of leading
            dimensions that should be mapped over.
        `out_axes`: Equivalent of `out_axes` of `jax.vmap`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dynamiqs.core._utils import _cartesian_vectorize
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> n = 2
        >>> x = jnp.ones((3, 4, 5, n, n))
        >>> y = jnp.ones((6, 7, n, n))
        >>> f = _cartesian_vectorize(func, (3, 3), 0)
        >>> f(x, y).shape
        (3, 4, 5, 6, 7, 2, 2)
    """
    # we use `jax.tree_util` to handle nested batching (such as `jump_ops`)
    leaves, treedef = jtu.tree_flatten(n_batch)

    # note: we apply the successive vmaps in reverse order, so the output
    # dimensions are in the correct order
    for i, leaf in reversed(list(enumerate(leaves))):
        if leaf > 0:
            # build the `in_axes` argument with the same structure as `n_batch`,
            # but with 0 at the `leaf` position
            in_axes = jtu.tree_map(lambda _: None, leaves)
            in_axes[i] = 0
            in_axes = jtu.tree_unflatten(treedef, in_axes)
            for _ in range(leaf):
                f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f
