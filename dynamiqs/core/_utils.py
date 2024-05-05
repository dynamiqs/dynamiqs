from __future__ import annotations

import functools

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

    @functools.wraps(func)
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
    f: callable, n_batch: PyTree[int], out_axes: PyTree[int | None]
) -> callable:
    # todo: write doc
    broadcast_shape = jtu.tree_leaves(n_batch, is_shape)
    broadcast_shape = jnp.broadcast_shapes(*broadcast_shape)
    in_axes = jtu.tree_map(
        lambda x: 0 if len(x) > 0 else None, n_batch, is_leaf=is_shape
    )

    for _ in range(len(broadcast_shape)):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    return f


def _cartesian_vectorize(
    f: callable, n_batch: PyTree[int], out_axes: PyTree[int | None]
) -> callable:
    # todo :write doc

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
    # Hamilonian. This prevents performing the Cartesian product
    # on all terms for the sum Hamiltonian.
    return _flat_vectorize(f, n_batch[:1] + (None,) * len(n_batch[1:]), out_axes)
