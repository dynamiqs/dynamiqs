from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src.lib import xla_client
from jaxtyping import ArrayLike, PyTree

from .._utils import cdtype, obj_type_str
from ..solver import Solver, _ODEAdaptiveStep
from ..time_array import (
    CallableTimeArray,
    ConstantTimeArray,
    ModulatedTimeArray,
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
                f'Argument must be an array-like or a time-array object, but has type'
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


def compute_vmap(
    f: callable,
    cartesian_batching: bool,
    is_batched: PyTree[bool],
    out_axes: PyTree[int | None],
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
                    f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
        else:
            # map over all batched dimensions at once
            in_axes = jax.tree_util.tree_map(lambda x: 0 if x else None, is_batched)
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
    elif isinstance(tarray, ModulatedTimeArray):
        return ModulatedTimeArray(
            False, False, tuple(arg.ndim > 0 for arg in tarray.args)
        )
    elif isinstance(tarray, CallableTimeArray):
        return CallableTimeArray(False, tuple(arg.ndim > 0 for arg in tarray.args))
    else:
        raise TypeError(f'Unsupported TimeArray type: {type(tarray).__name__}')
