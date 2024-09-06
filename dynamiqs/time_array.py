from __future__ import annotations

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

__all__ = ['constant', 'pwc', 'modulated', 'timecallable', 'TimeArray']


def constant(array: ArrayLike) -> ConstantTimeArray:
    r"""Instantiate a constant time-array.

    A constant time-array is defined by $O(t) = O_0$ for any time $t$, where $O_0$ is a
    constant array.

    Args:
        array _(array_like of shape (..., n, n))_: Constant array $O_0$.

    Returns:
        _(time-array object of shape (..., n, n) when called)_ Callable object
            returning $O_0$ for any time $t$.

    Examples:
        >>> H = dq.constant(dq.sigmaz())
        >>> H(0.0)
        Array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]], dtype=complex64)
        >>> H(1.0)
        Array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]], dtype=complex64)
    """
    array = jnp.asarray(array, dtype=cdtype())
    check_shape(array, 'array', '(..., n, n)')
    return ConstantTimeArray(array)


def pwc(times: ArrayLike, values: ArrayLike, array: ArrayLike) -> PWCTimeArray:
    r"""Instantiate a piecewise constant (PWC) time-array.

    A PWC time-array takes constant values over some time intervals. It is defined by
    $$
        O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
    $$
    where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular
    window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and
    $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant array.

    Note:
        The argument `times` must be sorted in ascending order, but does not
        need to be evenly spaced.

    Note:
        If the returned time-array is called for a time $t$ which does not belong to any
        time intervals, the returned array is null.

    Args:
        times _(array_like of shape (N+1,))_: Time points $t_k$ defining the boundaries
            of the time intervals, where _N_ is the number of time intervals.
        values _(array_like of shape (..., N))_: Constant values $c_k$ for each time
            interval.
        array _(array_like of shape (n, n))_: Constant array $O_0$.

    Returns:
        _(time-array object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> times = [0.0, 1.0, 2.0]
        >>> values = [3.0, -2.0]
        >>> array = dq.sigmaz()
        >>> H = dq.pwc(times, values, array)
        >>> H(-0.5)
        Array([[ 0.+0.j,  0.+0.j],
               [ 0.+0.j, -0.+0.j]], dtype=complex64)
        >>> H(0.0)
        Array([[ 3.+0.j,  0.+0.j],
               [ 0.+0.j, -3.+0.j]], dtype=complex64)
        >>> H(0.5)
        Array([[ 3.+0.j,  0.+0.j],
               [ 0.+0.j, -3.+0.j]], dtype=complex64)
        >>> H(1.0)
        Array([[-2.+0.j, -0.+0.j],
               [-0.+0.j,  2.-0.j]], dtype=complex64)
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

    # array
    array = jnp.asarray(array, dtype=cdtype())
    check_shape(array, 'array', '(n, n)')

    return PWCTimeArray(times, values, array)


def modulated(
    f: callable[[float], Scalar | Array],
    array: ArrayLike,
    *,
    discontinuity_ts: ArrayLike | None = None,
) -> ModulatedTimeArray:
    r"""Instantiate a modulated time-array.

    A modulated time-array is defined by $O(t) = f(t) O_0$ where $f(t)$ is a
    time-dependent scalar. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> Scalar | Array` that returns a scalar or an array of
    shape _(...)_ for any time $t$.

    Args:
        f _(function returning scalar or array of shape (...))_: Function with signature
            `f(t: float) -> Scalar | Array` that returns the modulating factor
            $f(t)$.
        array _(array_like of shape (n, n))_: Constant array $O_0$.
        discontinuity_ts _(array_like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
        _(time-array object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
        >>> H = dq.modulated(f, dq.sigmax())
        >>> H(0.5)
        Array([[-0.+0.j, -1.+0.j],
               [-1.+0.j, -0.+0.j]], dtype=complex64)
        >>> H(1.0)
        Array([[0.+0.j, 1.+0.j],
               [1.+0.j, 0.+0.j]], dtype=complex64)
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # array
    array = jnp.asarray(array, dtype=cdtype())
    check_shape(array, 'array', '(n, n)')

    # discontinuity_ts
    if discontinuity_ts is not None:
        discontinuity_ts = jnp.asarray(discontinuity_ts)
        discontinuity_ts = jnp.sort(discontinuity_ts)

    # make f a valid PyTree that is vmap-compatible
    f = BatchedCallable(f)

    return ModulatedTimeArray(f, array, discontinuity_ts)


def timecallable(
    f: callable[[float], Array], *, discontinuity_ts: ArrayLike | None = None
) -> CallableTimeArray:
    r"""Instantiate a callable time-array.

    A callable time-array is defined by $O(t) = f(t)$ where $f(t)$ is a
    time-dependent operator. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> Array` that returns an array of shape
    _(..., n, n)_ for any time $t$.

    Warning: The function `f` must return a JAX array (not an array-like object!)
        An error is raised if the function `f` does not return a JAX array. This error
        concerns any other array-like objects. This is enforced to avoid costly
        conversions at every time step of the numerical integration.

    Args:
        f _(function returning array of shape (..., n, n))_: Function with signature
            `(t: float) -> Array` that returns the array $f(t)$.
        discontinuity_ts _(array_like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
       _(time-array object of shape (..., n, n) when called)_ Callable object
            returning $O(t)$ for any time $t$.

    Examples:
        >>> f = lambda t: jnp.array([[t, 0], [0, 1 - t]])
        >>> H = dq.timecallable(f)
        >>> H(0.5)
        Array([[0.5, 0. ],
               [0. , 0.5]], dtype=float32)
        >>> H(1.0)
        Array([[1., 0.],
               [0., 0.]], dtype=float32)
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

    return CallableTimeArray(f, discontinuity_ts)


class Shape(tuple):
    """Helper class to help with Pytree handling."""


class TimeArray(eqx.Module):
    r"""Base class for time-dependent arrays.

    A time-array is a callable object that returns a JAX array for any time $t$. It is
    used to define time-dependent operators for Dynamiqs solvers.

    Attributes:
        dtype _(numpy.dtype)_: Data type.
        shape _(tuple of int)_: Shape.
        mT _(TimeArray)_: Returns the time-array transposed over its last two
            dimensions.
        ndim _(int)_: Number of dimensions.
        discontinuity_ts _(Array | None)_: Times at which there is a discontinuous jump
            in the time-array values (the array is always sorted, but does not
            necessarily contain unique values).

    Note: Arithmetic operation support
        Time-arrays support elementary operations:

        - negation (`__neg__`),
        - left-and-right element-wise addition/subtraction with other arrays or
            time-arrays (`__add__`, `__radd__`, `__sub__`, `__rsub__`),
        - left-and-right element-wise multiplication with other arrays (`__mul__`,
            `__rmul__`).
    """

    # Subclasses should implement:
    # - the properties: dtype, shape, mT, in_axes, discontinuity_ts
    # - the methods: reshape, broadcast_to, conj, __call__, __mul__, __add__

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Array`, `ConstantTimeArray` and the subclass type itself.

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def mT(self) -> TimeArray:
        pass

    @property
    @abstractmethod
    def in_axes(self) -> PyTree[int]:
        # returns the `in_axes` arguments that should be passed to vmap in order
        # to vmap the TimeArray correctly
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        # must be sorted, not necessarily unique values
        pass

    @abstractmethod
    def reshape(self, *shape: int) -> TimeArray:
        """Returns a reshaped copy of a time-array.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New time-array object with the given shape.
        """

    @abstractmethod
    def broadcast_to(self, *shape: int) -> TimeArray:
        """Broadcasts a time-array to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New time-array object with the given shape.
        """

    @abstractmethod
    def conj(self) -> TimeArray:
        """Returns the element-wise complex conjugate of the time-array.

        Returns:
            New time-array object with element-wise complex conjuguated values.
        """

    def squeeze(self, axis: int | None = None) -> TimeArray:
        """Squeeze a time-array.

        Args:
            axis: Axis to squeeze. If `none`, all axes with dimension 1 are squeezed.

        Returns:
            New time-array object with squeezed_shape
        """
        if axis is None:
            shape = self.shape
            x = self
            for i, s in reversed(list(enumerate(shape))):
                if s == 1:
                    x = x.squeeze(i)
            return x

        if axis >= self.ndim:
            raise ValueError(
                f'Cannot squeeze axis {axis} from a time-array with {self.ndim} axes.'
            )
        return self.reshape(*self.shape[:axis], *self.shape[axis + 1 :])

    @abstractmethod
    def __call__(self, t: ScalarLike) -> Array:
        """Returns the time-array evaluated at a given time.

        Args:
            t: Time at which to evaluate the array.

        Returns:
            Array evaluated at time $t$.
        """

    def __neg__(self) -> TimeArray:
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: ArrayLike) -> TimeArray:
        pass

    def __rmul__(self, y: ArrayLike) -> TimeArray:
        return self * y

    @abstractmethod
    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        pass

    def __radd__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return self + y

    def __sub__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return self + (-y)

    def __rsub__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return y + (-self)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'


class ConstantTimeArray(TimeArray):
    array: Array

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def mT(self) -> TimeArray:
        return ConstantTimeArray(self.array.mT)

    @property
    def in_axes(self) -> PyTree[int]:
        return ConstantTimeArray(Shape(self.array.shape[:-2]))

    @property
    def discontinuity_ts(self) -> Array | None:
        return None

    def reshape(self, *shape: int) -> TimeArray:
        return ConstantTimeArray(self.array.reshape(*shape))

    def broadcast_to(self, *shape: int) -> TimeArray:
        return ConstantTimeArray(jnp.broadcast_to(self.array, shape))

    def conj(self) -> TimeArray:
        return ConstantTimeArray(self.array.conj())

    def __call__(self, t: ScalarLike) -> Array:  # noqa: ARG002
        return self.array

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ConstantTimeArray(self.array * y)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            return ConstantTimeArray(jnp.asarray(other, dtype=cdtype()) + self.array)
        elif isinstance(other, ConstantTimeArray):
            return ConstantTimeArray(self.array + other.array)
        elif isinstance(other, TimeArray):
            return SummedTimeArray([self, other])
        else:
            return NotImplemented


class PWCTimeArray(TimeArray):
    times: Array  # (nv+1,)
    values: Array  # (..., nv)
    array: Array  # (n, n)

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.values.shape[:-1], *self.array.shape

    @property
    def mT(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values, self.array.mT)

    @property
    def in_axes(self) -> PyTree[int]:
        return PWCTimeArray(Shape(), Shape(self.values.shape[:-1]), Shape())

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.times

    def reshape(self, *shape: int) -> TimeArray:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = self.values.reshape(*shape)
        return PWCTimeArray(self.times, values, self.array)

    def broadcast_to(self, *shape: int) -> TimeArray:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = jnp.broadcast_to(self.values, shape)
        return PWCTimeArray(self.times, values, self.array)

    def conj(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values.conj(), self.array.conj())

    def prefactor(self, t: ScalarLike) -> Array:
        def _zero(_: float) -> Array:
            return jnp.zeros_like(self.values[..., 0])  # (...)

        def _pwc(t: float) -> Array:
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

        return lax.cond(
            jnp.logical_or(t < self.times[0], t >= self.times[-1]), _zero, _pwc, t
        )

    def __call__(self, t: ScalarLike) -> Array:
        return self.prefactor(t)[..., None, None] * self.array

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return PWCTimeArray(self.times, self.values, self.array * y)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([self, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([self, other])
        else:
            return NotImplemented


class ModulatedTimeArray(TimeArray):
    f: BatchedCallable  # (...)
    array: Array  # (n, n)
    _disc_ts: Array | None

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.f.shape, *self.array.shape

    @property
    def mT(self) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array.mT, self._disc_ts)

    @property
    def in_axes(self) -> PyTree[int]:
        return ModulatedTimeArray(Shape(self.f.shape), Shape(), Shape())

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeArray:
        f = self.f.reshape(*shape[:-2])
        return ModulatedTimeArray(f, self.array, self._disc_ts)

    def broadcast_to(self, *shape: int) -> TimeArray:
        f = self.f.broadcast_to(*shape[:-2])
        return ModulatedTimeArray(f, self.array, self._disc_ts)

    def conj(self) -> TimeArray:
        f = self.f.conj()
        return ModulatedTimeArray(f, self.array.conj(), self._disc_ts)

    def prefactor(self, t: ScalarLike) -> Array:
        return self.f(t)

    def __call__(self, t: ScalarLike) -> Array:
        return self.prefactor(t)[..., None, None] * self.array

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array * y, self._disc_ts)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([self, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([self, other])
        else:
            return NotImplemented


class CallableTimeArray(TimeArray):
    f: BatchedCallable  # (..., n, n)
    _disc_ts: Array | None

    @property
    def dtype(self) -> jnp.dtype:
        return self.f.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f.shape

    @property
    def mT(self) -> TimeArray:
        f = jtu.Partial(lambda t: self.f(t).mT)
        return CallableTimeArray(f, self._disc_ts)

    @property
    def in_axes(self) -> PyTree[int]:
        return CallableTimeArray(Shape(self.f.shape[:-2]), Shape())

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeArray:
        f = self.f.reshape(*shape)
        return CallableTimeArray(f, self._disc_ts)

    def broadcast_to(self, *shape: int) -> TimeArray:
        f = self.f.broadcast_to(*shape)
        return CallableTimeArray(f, self._disc_ts)

    def conj(self) -> TimeArray:
        f = self.f.conj()
        return CallableTimeArray(f, self._disc_ts)

    def __call__(self, t: ScalarLike) -> Array:
        return self.f(t)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        f = self.f * y
        return CallableTimeArray(f, self._disc_ts)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([self, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([self, other])
        else:
            return NotImplemented


class SummedTimeArray(TimeArray):
    timearrays: list[TimeArray]

    @property
    def dtype(self) -> jnp.dtype:
        return self.timearrays[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(*[tarray.shape for tarray in self.timearrays])

    @property
    def mT(self) -> TimeArray:
        timearrays = [tarray.mT for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    @property
    def in_axes(self) -> PyTree[int]:
        timearrays = [tarray.in_axes for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [tarray.discontinuity_ts for tarray in self.timearrays]
        return _concatenate_sort(*ts)

    def reshape(self, *shape: int) -> TimeArray:
        timearrays = [tarray.reshape(*shape) for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    def broadcast_to(self, *shape: int) -> TimeArray:
        timearrays = [tarray.broadcast_to(*shape) for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    def conj(self) -> TimeArray:
        timearrays = [tarray.conj() for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    def __call__(self, t: ScalarLike) -> Array:
        return jax.tree_util.tree_reduce(
            jnp.add, [tarray(t) for tarray in self.timearrays]
        )

    def __mul__(self, y: ArrayLike) -> TimeArray:
        timearrays = [tarray * y for tarray in self.timearrays]
        return SummedTimeArray(timearrays)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([*self.timearrays, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([*self.timearrays, other])
        else:
            return NotImplemented


class BatchedCallable(eqx.Module):
    # this class turns a callable into a PyTree that is vmap-compatible

    f: callable[[float], Array]
    indices: list[Array]

    def __init__(self, f: callable[[float], Scalar | Array]):
        # make f a valid PyTree with `Partial` and convert its output to an array
        self.f = jtu.Partial(lambda t: jnp.asarray(f(t)))
        shape = jax.eval_shape(f, 0.0).shape
        self.indices = list(jnp.indices(shape))

    def __call__(self, t: ScalarLike) -> Array:
        return self.f(t)[tuple(self.indices)]

    @property
    def dtype(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).shape

    def reshape(self, *shape: tuple[int, ...]) -> BatchedCallable:
        f = lambda t: self.f(t).reshape(shape)
        return BatchedCallable(f)

    def broadcast_to(self, *shape: tuple[int, ...]) -> BatchedCallable:
        f = lambda t: jnp.broadcast_to(self.f(t), shape)
        return BatchedCallable(f)

    def conj(self) -> BatchedCallable:
        f = lambda t: self.f(t).conj()
        return BatchedCallable(f)

    def squeeze(self, i: int) -> BatchedCallable:
        f = lambda t: jnp.squeeze(self.f(t), i)
        return BatchedCallable(f)

    def __mul__(self, y: ArrayLike) -> BatchedCallable:
        f = lambda t: self.f(t) * y
        return BatchedCallable(f)
