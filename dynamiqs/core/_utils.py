from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.lib import xla_client
from jaxtyping import ArrayLike, PyTree

from .._utils import cdtype, obj_type_str
from ..solver import Solver, _ODEAdaptiveStep
from ..time_array import ConstantTimeArray, Shape, TimeArray
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


def catch_xla_runtime_error(func: callable) -> callable:
    # Decorator to catch `XlaRuntimeError`` exceptions, and set a more friendly
    # exception message. Note that this will not work for jitted function, as the
    # exception code will be traced out.

    @ft.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        try:
            return func(*args, **kwargs)
        except xla_client.XlaRuntimeError as e:
            # === `max_steps` reached error
            eqx_max_steps_error_msg = (
                'EqxRuntimeError: The maximum number of solver steps was reached. '
            )
            if eqx_max_steps_error_msg in str(e):
                default_max_steps = _ODEAdaptiveStep.max_steps
                raise RuntimeError(
                    'The maximum number of solver steps has been reached (the default'
                    f' value is `max_steps={default_max_steps:_}`). Try increasing'
                    ' `max_steps` with the `solver` argument, e.g.'
                    ' `solver=dq.solver.Tsit5(max_steps=1_000_000)`.'
                ) from e
            # === other errors
            raise RuntimeError(
                'An internal JAX error interrupted the execution, please report this to'
                ' the dynamiqs developers by opening an issue on GitHub or sending a'
                ' message on dynamiqs Slack (links available at'
                ' https://www.dynamiqs.org/getting_started/lets-talk.html).'
            ) from e

    return wrapper


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


def is_shape(x: object) -> bool:
    return isinstance(x, Shape)


def _flat_vectorize(
    f: callable, n_batch: PyTree[int | False], out_axes: PyTree[int | False]
) -> callable:
    # todo: write doc
    n_batch = tree_false_to_none(n_batch, is_leaf=is_shape)
    out_axes = tree_false_to_none(out_axes)

    broadcast_shape = jtu.tree_leaves(n_batch, is_shape)
    broadcast_shape = jnp.broadcast_shapes(*broadcast_shape)
    in_axes = jtu.tree_map(
        lambda x: 0 if len(x) > 0 else None, n_batch, is_leaf=is_shape
    )

    for _ in range(len(broadcast_shape)):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f


def tree_false_to_none(
    tree: PyTree, is_leaf: callable[PyTree, bool] | None = None
) -> PyTree:
    return jtu.tree_map(lambda x: x if x is not False else None, tree, is_leaf=is_leaf)


def _h_vectorize(  # noqa: C901
    f: TimeArray, n_batch_false: PyTree, out_axes_false: PyTree
) -> TimeArray:
    """Vectorize a Hamiltonian function.

    Args:
        f: the Hamiltonian function.
        n_batch_false: the batch shape of the Hamiltonian.
        out_axes_false: the out axes of the Hamiltonian.
    """
    # JAX completely dismisses leaves with a `None` when applying the `tree_map`, so we
    # need to keep one version of the batch shape with `False` instead of `None`
    # to keep the structure.
    n_batch = tree_false_to_none(n_batch_false, is_leaf=is_shape)
    out_axes = tree_false_to_none(out_axes_false)

    broadcast_shape = jtu.tree_leaves(n_batch, is_shape)
    broadcast_shape = jnp.broadcast_shapes(*broadcast_shape)

    def tree_map_fn(i: int, x: tuple[int, ...]) -> int | None:
        """Args:
            i: the index in `broadcast_shape`.
            x: the shape of the leaf.

        Returns:
            The in axes for the vmap function.
        """
        if len(x) <= i or x[len(x) - i - 1] == 1:
            return None
        else:
            return 0

    n = len(broadcast_shape)
    expand_dims = []  # dimensions '1' that will be lost during the vmap but that
    # we want to keep in the results.
    for i in range(len(broadcast_shape)):
        in_axes = jtu.tree_map(ft.partial(tree_map_fn, i), n_batch, is_leaf=is_shape)
        if jtu.tree_all(jtu.tree_map(lambda x: x is None, in_axes)):
            expand_dims.append(n - i - 1)
        else:
            f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    expand_dims = sorted(expand_dims)

    def squeeze_args(size: PyTree, arg: PyTree) -> PyTree:
        """Squeeze the all the arguments with a dimension 1."""
        if is_shape(size):
            for i, s in reversed(list(enumerate(size))):
                if s == 1:
                    arg = arg.squeeze(i)

        return arg

    def unsqueeze_args(out_ax: PyTree, result: PyTree) -> PyTree:
        """Unsqueeze the result."""
        if out_ax is not False:
            for dim in expand_dims:
                result = jtu.tree_map(
                    ft.partial(lambda t, dim: jnp.expand_dims(t, dim), dim=dim), result
                )

        return result

    def wrap(*args: [PyTree]) -> PyTree:
        squeezed_args = jtu.tree_map(squeeze_args, n_batch_false, args)
        result = f(*squeezed_args)
        return jtu.tree_map(unsqueeze_args, out_axes_false, result)

    return wrap


def _cartesian_vectorize(
    f: callable, n_batch: PyTree[int], out_axes: PyTree[int | None]
) -> callable:
    # todo :write doc
    n_batch = tree_false_to_none(n_batch, is_leaf=is_shape)
    out_axes = tree_false_to_none(out_axes)

    # We use `jax.tree_util` to handle nested batching (such as `jump_ops`).
    # Only the second to last batch terms are taken into account in order to
    # have proper batching of SummedTimeArrays (see below).
    leaves, treedef = jtu.tree_flatten((None,) + n_batch[1:], is_leaf=is_shape)

    # note: we apply the successive vmaps in reverse order, so the output
    # dimensions are in the correct order
    for i, leaf in reversed(list(enumerate(leaves))):
        leaf_len = len(leaf)
        if leaf_len > 0:
            # build the `in_axes` argument with the same structure as `n_batch`,
            # but with 0 at the `leaf` position
            in_axes = jtu.tree_map(lambda _: None, leaves)
            in_axes[i] = 0
            in_axes = jtu.tree_unflatten(treedef, in_axes)
            for _ in range(leaf_len):
                f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    # We flat vectorize on the first n_batch term, which is the
    # Hamiltonian. This prevents performing the Cartesian product
    # on all terms for the sum Hamiltonian.
    return _h_vectorize(f, n_batch[:1] + (False,) * len(n_batch[1:]), out_axes)
