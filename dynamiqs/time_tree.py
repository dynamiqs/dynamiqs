from __future__ import annotations

import functools as ft
from abc import abstractmethod
from typing import get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, lax
from jaxtyping import ArrayLike, PyTree, Scalar, ScalarLike

from ._checks import check_shape, check_times
from ._utils import _concatenate_sort, cdtype, obj_type_str
from .qarrays.layout import Layout, promote_layouts
from .qarrays.qarray import QArray, QArrayLike, isqarraylike
from .qarrays.utils import asqarray

__all__ = ['TimeTree', 'constant', 'modulated', 'pwc', 'timecallable']


def constant(tree: PyTree) -> ConstantTimeTree:
    r"""Instantiate a constant time-tree.

    A constant time-tree is defined by $O(t) = O_0$ for any time $t$, where $O_0$ is a
    constant qarray.

    Args:
        qarray _(qarray_like of shape (..., n, n))_: Constant qarray $O_0$.

    Returns:
        _(time-tree object of shape (..., n, n) when called)_ Callable object
            returning $O_0$ for any time $t$.

    Examples:
        >>> H = dq.constant(dq.sigmaz())
        >>> H(0.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅   ]
         [   ⋅    -1.+0.j]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅   ]
         [   ⋅    -1.+0.j]]
    """
    check_shape(tree, 'tree', '(..., n, n)')
    return ConstantTimeTree(tree)


def pwc(times: ArrayLike, values: ArrayLike, tree: PyTree) -> PWCTimeTree:
    r"""Instantiate a piecewise constant (PWC) time-tree.

    A PWC time-tree takes constant values over some time intervals. It is defined by
    $$
        O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
    $$
    where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular
    window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and
    $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant qarray.

    Note:
        The argument `times` must be sorted in ascending order, but does not
        need to be evenly spaced.

    Note:
        If the returned time-tree is called for a time $t$ which does not belong to
        any time intervals, the returned qarray is null.

    Args:
        times _(array_like of shape (N+1,))_: Time points $t_k$ defining the boundaries
            of the time intervals, where _N_ is the number of time intervals.
        values _(array_like of shape (..., N))_: Constant values $c_k$ for each time
            interval.
        qarray _(qarray_like of shape (n, n))_: Constant qarray $O_0$.

    Returns:
        _(time-tree object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> times = [0.0, 1.0, 2.0]
        >>> values = [3.0, -2.0]
        >>> qarray = dq.sigmaz()
        >>> H = dq.pwc(times, values, qarray)
        >>> H(-0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅   ]
         [  ⋅      ⋅   ]]
        >>> H(0.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 3.+0.j    ⋅   ]
         [   ⋅    -3.+0.j]]
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 3.+0.j    ⋅   ]
         [   ⋅    -3.+0.j]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[-2.+0.j    ⋅   ]
         [   ⋅     2.+0.j]]
    """
    # times
    times = jnp.asarray(times)
    times = check_times(times, 'times')

    # values
    values = jnp.asarray(values, dtype=cdtype())
    if values.shape[-1] != len(times) - 1:
        raise TypeError(
            'Argument `values` must have shape `(..., len(times)-1)`, but has shape'
            f' `{values.shape}.'
        )

    # qarray
    tree = asqarray(tree)
    check_shape(tree, 'tree', '(n, n)')

    return PWCTimeTree(times, values, tree)


def modulated(
        f: callable[[float], Scalar | Array],
        tree: PyTree,
        *,
        discontinuity_ts: ArrayLike | None = None,
) -> ModulatedTimeTree:
    r"""Instantiate a modulated time-tree.

    A modulated time-tree is defined by $O(t) = f(t) O_0$ where $f(t)$ is a
    time-dependent scalar. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> Scalar | Array` that returns a scalar or an array of
    shape _(...)_ for any time $t$.

    Args:
        f _(function returning scalar or array of shape (...))_: Function with signature
            `f(t: float) -> Scalar | Array` that returns the modulating factor
            $f(t)$.
        qarray _(qarray_like of shape (n, n))_: Constant qarray $O_0$.
        discontinuity_ts _(array_like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
        _(time-tree object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
        >>> H = dq.modulated(f, dq.sigmax())
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[   ⋅    -1.+0.j]
         [-1.+0.j    ⋅   ]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[  ⋅    1.+0.j]
         [1.+0.j   ⋅   ]]
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # qarray
    tree = asqarray(tree)
    check_shape(tree, 'tree', '(n, n)')

    # discontinuity_ts
    if discontinuity_ts is not None:
        discontinuity_ts = jnp.asarray(discontinuity_ts)
        discontinuity_ts = jnp.sort(discontinuity_ts)

    # make f a valid PyTree that is vmap-compatible
    f = BatchedCallable(f)

    return ModulatedTimeTree(f, tree, discontinuity_ts)


def timecallable(
        f: callable[[float], PyTree], *, discontinuity_ts: ArrayLike | None = None
) -> CallableTimeTree:
    r"""Instantiate a callable time-tree.

    A callable time-tree is defined by $O(t) = f(t)$ where $f(t)$ is a
    time-dependent operator. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> PyTree` that returns a PyTree of shape _(..., n, n)_
    for any time $t$.

    Args:
        f _(function returning qarray of shape (..., n, n))_: Function with
            signature `(t: float) -> QArray` that returns the qarray $f(t)$.
        discontinuity_ts _(array_like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
       _(time-tree object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> f = lambda t: dq.asqarray([[t, 0], [0, 1 - t]])
        >>> H = dq.timecallable(f)
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
        [[0.5 0. ]
         [0.  0.5]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
        [[1. 0.]
         [0. 0.]]
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # discontinuity_ts
    if discontinuity_ts is not None:
        discontinuity_ts = jnp.asarray(discontinuity_ts)
        discontinuity_ts = jnp.sort(discontinuity_ts)

    # make f a valid PyTree that is vmap-compatible
    f = BatchedCallable(f)

    return CallableTimeTree(f, discontinuity_ts)


class Shape(tuple):
    """Helper class to help with Pytree handling."""


class TimeTree(eqx.Module):
    r"""Base class for time-dependent PyTrees.

    A time-tree is a callable object that returns a PyTree for any time $t$. It is
    used to define time-dependent operators for Dynamiqs solvers.

    Attributes:
        discontinuity_ts _(Array | None)_: Times at which there is a discontinuous jump
            in the time-tree values (the array is always sorted, but does not
            necessarily contain unique values).

    Note: Arithmetic operation support
        time-trees support elementary operations:

        - negation (`__neg__`),
        - left-and-right element-wise addition/subtraction with other arrays, PyTrees or
            time-trees (`__add__`, `__radd__`, `__sub__`, `__rsub__`),
        - left-and-right element-wise multiplication with other arrays (`__mul__`,
            `__rmul__`).
    """

    # Subclasses should implement:
    # - the properties: discontinuity_ts
    # - the methods: reshape, broadcast_to, conj, __call__, __mul__, __add__

    def __getattr__(self, name):
        try:
            attr = super().__getattr__(name)
        except AttributeError:
            attr = getattr(jax.eval_shape(self.__call__, 0.0), name)

        if callable(attr):
            f = self._map_callable(attr)
            return f
        else:
            return attr

    @abstractmethod
    def _map_callable(self, fun):
        pass

    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        # must be sorted, not necessarily unique values
        pass

    @abstractmethod
    def broadcast_to(self, *shape: int) -> TimeTree:
        """Broadcasts a time-tree to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New time-tree object with the given shape.
        """

    @abstractmethod
    def __call__(self, t: ScalarLike) -> QArray:
        """Returns the time-tree evaluated at a given time.

        Args:
            t: Time at which to evaluate the time-tree.

        Returns:
            QArray evaluated at time $t$.
        """

    def __neg__(self) -> TimeTree:
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: QArrayLike) -> TimeTree:
        pass

    def __rmul__(self, y: QArrayLike) -> TimeTree:
        return self * y

    @abstractmethod
    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        pass

    def __radd__(self, y: QArrayLike | TimeTree) -> TimeTree:
        return self + y

    def __sub__(self, y: QArrayLike | TimeTree) -> TimeTree:
        return self + (-y)

    def __rsub__(self, y: QArrayLike | TimeTree) -> TimeTree:
        return y + (-self)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, '
            f'layout={self.layout})'
        )


class ConstantTimeTree(TimeTree):
    tree: PyTree

    def _map_callable(self, fun):
        def res(*args, **kwargs):
            f = ft.partial(fun.__func__, *args, **kwargs)
            return jax.tree.map(f, self, is_leaf=lambda x: isinstance(x, QArray))
        return res

    @property
    def in_axes(self) -> PyTree[int | None]:
        return ConstantTimeTree(0)

    @property
    def discontinuity_ts(self) -> Array | None:
        return None

    def reshape(self, *shape: int) -> TimeTree:
        return ConstantTimeTree(self.tree.reshape(*shape))

    def broadcast_to(self, *shape: int) -> TimeTree:
        if isinstance(self.tree, QArray):
            res = self.tree.broadcast_to(*shape)
        else:
            res = jnp.broadcast_to(self.tree, shape)

        return ConstantTimeTree(res)

    def conj(self) -> TimeTree:
        return ConstantTimeTree(self.tree.conj())

    def __call__(self, t: ScalarLike) -> QArray:  # noqa: ARG002
        return self.tree

    def __mul__(self, y: QArrayLike) -> TimeTree:
        return ConstantTimeTree(self.tree * y)

    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        if isqarraylike(y):
            return ConstantTimeTree(asqarray(y) + self.tree)
        elif isinstance(y, ConstantTimeTree):
            return ConstantTimeTree(self.tree + y.tree)
        elif isinstance(y, TimeTree):
            return SummedTimeTree([self, y])
        else:
            return NotImplemented


class PWCTimeTree(TimeTree):
    times: Array  # (nv+1,)
    values: Array  # (..., nv)
    tree: PyTree  # (n, n)

    def _map_callable(self, fun):
        def res(*args, **kwargs):
            # f1 = ft.partial(fun.__func__, *args, **kwargs)
            f1 = fun.__func__
            # tree = jax.tree.map(self.tree, f1, is_leaf=lambda x: isinstance(x, QArray))
            tree = f1(self.tree)
            return PWCTimeTree(self.times, self.values, tree)
        return res

    @property
    def in_axes(self) -> PyTree[int | None]:
        return PWCTimeTree(None, 0, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.times

    def reshape(self, *shape: int) -> TimeTree:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = self.values.reshape(*shape)
        return PWCTimeTree(self.times, values, self.tree)

    def broadcast_to(self, *shape: int) -> TimeTree:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = jnp.broadcast_to(self.values, shape)
        return PWCTimeTree(self.times, values, self.tree)

    def conj(self) -> TimeTree:
        return PWCTimeTree(self.times, self.values.conj(), self.tree.conj())

    def prefactor(self, t: ScalarLike) -> Array:
        def _zero(_: float) -> Array:
            return jnp.zeros_like(self.values[..., 0])  # (...)

        def _pwc(t: float) -> Array:
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

        return lax.cond(
            jnp.logical_or(t < self.times[0], t >= self.times[-1]), _zero, _pwc, t
        )

    def __call__(self, t: ScalarLike) -> QArray:
        return self.prefactor(t)[..., None, None] * self.tree

    def __mul__(self, y: QArrayLike) -> TimeTree:
        return PWCTimeTree(self.times, self.values, self.tree * y)

    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        if isqarraylike(y):
            y = ConstantTimeTree(asqarray(y))
            return SummedTimeTree([self, y])
        elif isinstance(y, TimeTree):
            return SummedTimeTree([self, y])
        else:
            return NotImplemented


class ModulatedTimeTree(TimeTree):
    f: BatchedCallable  # (...)
    tree: PyTree
    _disc_ts: Array | None

    def _map_callable(self, fun):
        def res(*args, **kwargs):
            f = ft.partial(fun, *args, **kwargs)
            tree = jax.tree.map(self.tree, f)
            return PWCTimeTree(self.f, tree, self._disc_ts)
        return res

    @property
    def in_axes(self) -> PyTree[int | None]:
        return ModulatedTimeTree(0, None, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeTree:
        f = self.f.reshape(*shape[:-2])
        return ModulatedTimeTree(f, self.tree, self._disc_ts)

    def broadcast_to(self, *shape: int) -> TimeTree:
        f = self.f.broadcast_to(*shape[:-2])
        return ModulatedTimeTree(f, self.tree, self._disc_ts)

    def conj(self) -> TimeTree:
        f = self.f.conj()
        return ModulatedTimeTree(f, self.tree.conj(), self._disc_ts)

    def prefactor(self, t: ScalarLike) -> Array:
        return self.f(t)

    def __call__(self, t: ScalarLike) -> QArray:
        return self.prefactor(t)[..., None, None] * self.tree

    def __mul__(self, y: QArrayLike) -> TimeTree:
        return ModulatedTimeTree(self.f, self.tree * y, self._disc_ts)

    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        if isqarraylike(y):
            y = ConstantTimeTree(asqarray(y))
            return SummedTimeTree([self, y])
        elif isinstance(y, TimeTree):
            return SummedTimeTree([self, y])
        else:
            return NotImplemented


class CallableTimeTree(TimeTree):
    f: BatchedCallable  # (..., n, n)
    _disc_ts: Array | None

    def _map_callable(self, fun):
        def res(*args, **kwargs):
            # f1 = lambda x: ft.partial(fun.__func__, x, *args, **kwargs)
            f1 = fun.__func__
            # f2 = lambda t: jax.tree.map(f1, self.f(t), is_leaf=lambda x: isinstance(x, QArray))
            f2 = lambda t: f1(self.f(t))
            return CallableTimeTree(f2, self._disc_ts)
        return res

    @property
    def mT(self) -> TimeTree:
        f = jtu.Partial(lambda t: self.f(t).mT)
        return CallableTimeTree(f, self._disc_ts)

    @property
    def in_axes(self) -> PyTree[int | None]:
        return CallableTimeTree(0, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeTree:
        f = self.f.reshape(*shape)
        return CallableTimeTree(f, self._disc_ts)

    def broadcast_to(self, *shape: int) -> TimeTree:
        f = self.f.broadcast_to(*shape)
        return CallableTimeTree(f, self._disc_ts)

    def __call__(self, t: ScalarLike) -> QArray:
        return self.f(t)

    def __mul__(self, y: QArrayLike) -> TimeTree:
        f = self.f * y
        return CallableTimeTree(f, self._disc_ts)

    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        if isinstance(y, get_args(ScalarLike)):
            return ConstantTimeTree(self.f + y)
        elif isqarraylike(y):
            y = ConstantTimeTree(asqarray(y))
            return SummedTimeTree([self, y])
        elif isinstance(y, TimeTree):
            return SummedTimeTree([self, y])
        else:
            return NotImplemented


class SummedTimeTree(TimeTree):
    time_trees: list[TimeTree]

    def __init__(self, time_trees: list[TimeTree], check: bool = True):
        if check:
            # verify all time-trees of the sum are broadcast compatible
            shape = jnp.broadcast_shapes(*[tt.shape for tt in time_trees])
            # ensure all time-trees can be jointly vmapped over (as specified by the
            # `in_axes` property)
            time_trees = [tt.broadcast_to(*shape) for tt in time_trees]
        self.time_trees = time_trees

    def _map_callable(self, fun):
        def res(*args, **kwargs):
            f = ft.partial(fun, *args,**kwargs)
            time_trees = jax.tree.map(f, self.time_trees)
            return SummedTimeTree(time_trees, self._disc_ts)
        return res

    @property
    def dtype(self) -> jnp.dtype:
        dtypes = [tt.dtype for tt in  self.time_trees]
        return ft.reduce(jnp.promote_types, dtypes)

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(*[tt.shape for tt in  self.time_trees])

    @property
    def layout(self) -> Layout:
        layouts = [tt.layout for tt in  self.time_trees]
        return ft.reduce(promote_layouts, layouts)

    @property
    def in_axes(self) -> PyTree[int | None]:
        in_axes_list = [tt.in_axes for tt in  self.time_trees]
        return SummedTimeTree(in_axes_list, check=False)

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [tt.discontinuity_ts for tt in  self.time_trees]
        return _concatenate_sort(*ts)

    def reshape(self, *shape: int) -> TimeTree:
        time_trees = [tt.reshape(*shape) for tt in  self.time_trees]
        return SummedTimeTree(time_trees)

    def broadcast_to(self, *shape: int) -> TimeTree:
        time_trees = [tt.broadcast_to(*shape) for tt in  self.time_trees]
        return SummedTimeTree(time_trees)

    def __call__(self, t: ScalarLike) -> QArray:
        return ft.reduce(
            lambda x, y: x + y, [tt(t) for tt in  self.time_trees]
        )

    def __mul__(self, y: QArrayLike) -> TimeTree:
        time_trees = [tt * y for tt in  self.time_trees]
        return SummedTimeTree(time_trees)

    def __add__(self, y: QArrayLike | TimeTree) -> TimeTree:
        if isqarraylike(y):
            y = ConstantTimeTree(asqarray(y))
            return SummedTimeTree([*self.time_trees, y])
        elif isinstance(y, TimeTree):
            return SummedTimeTree([*self.time_trees, y])
        else:
            return NotImplemented


class BatchedCallable(eqx.Module):
    # this class turns a callable into a PyTree that is vmap-compatible

    f: callable[[float], QArrayLike]
    indices: list[Array]

    def __init__(self, f: callable[[float], QArrayLike]):
        # make f a valid PyTree with `Partial` and convert its output to a qarray
        self.f = jtu.Partial(f)
        eval_shape = jax.eval_shape(f, 0.0)
        if isinstance(eval_shape, QArray):
            shape = eval_shape.shape[:-2]
        else:
            shape = eval_shape.shape
        self.indices = list(jnp.indices(shape))

    def __call__(self, t: ScalarLike) -> QArrayLike:
        if len(self.indices) == 0:
            return self.f(t)
        else:
            return self.f(t)[tuple(self.indices)]

    @property
    def dtype(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).shape

    @property
    def layout(self) -> Layout:
        return jax.eval_shape(self.f, 0.0).layout

    def reshape(self, *shape: tuple[int, ...]) -> BatchedCallable:
        f = lambda t: self.f(t).reshape(shape)
        return BatchedCallable(f)

    def broadcast_to(self, *shape: tuple[int, ...]) -> BatchedCallable:
        def f(t: float) -> QArrayLike:
            res = self.f(t)
            if isinstance(res, QArray):
                return res.broadcast_to(*shape)
            else:
                return jnp.broadcast_to(res, shape)

        return BatchedCallable(f)

    def conj(self) -> BatchedCallable:
        f = lambda t: self.f(t).conj()
        return BatchedCallable(f)

    def squeeze(self, i: int) -> BatchedCallable:
        f = lambda t: jnp.squeeze(self.f(t), i)
        return BatchedCallable(f)

    def __add__(self, y: ScalarLike) -> BatchedCallable:
        return BatchedCallable(lambda t: self.f(t) + y)

    def __mul__(self, y: ArrayLike) -> BatchedCallable:
        f = lambda t: self.f(t) * y
        return BatchedCallable(f)
