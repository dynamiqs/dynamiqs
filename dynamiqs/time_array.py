from __future__ import annotations

from abc import abstractmethod
from typing import get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import Partial
from jaxtyping import ArrayLike, PyTree, ScalarLike

from ._checks import check_shape, check_times
from ._utils import cdtype, expand_as_broadcastable, obj_type_str

__all__ = ['constant', 'pwc', 'modulated', 'timecallable', 'TimeArray']


def constant(array: ArrayLike) -> ConstantTimeArray:
    r"""Instantiate a constant time-array.

    A constant time-array is defined by $O(t) = O_0$ for any time $t$, where $O_0$ is a
    constant array.

    Args:
        array _(array_like of shape (..., n, n))_: Constant array $O_0$.

    Returns:
        _(time-array object)_ Callable object returning $O_0$ for any time $t$.
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

    Notes:
        The argument `times` argument must be sorted in ascending order, but does not
        need to be evenly spaced.

    Notes:
        If the returned time-array is called for a time $t$ which does not belong to any
        time intervals, the returned array is null.

    Args:
        times _(array_like of shape (N+1,))_: Time points $t_k$ defining the boundaries
            of the time intervals, where _N_ is the number of time intervals.
        values _(array_like of shape (..., N))_: Constant values $c_k$ for each time
            interval.
        array _(array_like of shape (n, n))_: Constant array $O_0$.

    Returns:
        _(time-array object)_ Callable object returning $O(t)$ for any time $t$.
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
    f: callable[[float, ...], Array], array: ArrayLike, *, args: tuple[PyTree] = ()
) -> ModulatedTimeArray:
    r"""Instantiate a modulated time-array.

    A modulated time-array is defined by $O(t) = f(t) O_0$ where $f(t)$ is a
    time-dependent scalar. The function $f$ is defined by passing a Python function
    with signature `f(t: float, *args: PyTree) -> Array` that returns an array of shape
    _(...)_ for any time $t$.

    Args:
        f _(function returning array of shape (...))_: Function with signature
            `f(t: float, *args: PyTree) -> Array` that returns the modulating factor
            $f(t)$.
        array _(array_like of shape (n, n))_: Constant array $O_0$.
        args: Other positional arguments passed to the function $f$.

    Returns:
        _(time-array object)_ Callable object returning $O(t)$ for any time $t$.
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # array
    array = jnp.asarray(array, dtype=cdtype())
    check_shape(array, 'array', '(n, n)')

    # Pass `f` through `jax.tree_util.Partial`.
    # This is necessary:
    # (1) to make f a Pytree, and
    # (2) to avoid jitting again every time the args change.
    f = Partial(f)

    return ModulatedTimeArray(f, array, args)


def timecallable(
    f: callable[[float, ...], Array], *, args: tuple[PyTree] = ()
) -> CallableTimeArray:
    r"""Instantiate a callable time-array.

    A callable time-array is defined by $O(t) = f(t)$ where $f(t)$ is a
    time-dependent operator. The function $f$ is defined by passing a Python function
    with signature `f(t: float, *args: PyTree) -> Array` that returns an array of shape
    _(..., n, n)_ for any time $t$.

    Args:
        f _(function returning array of shape (..., n, n))_: Function with signature
            `(t: float, *args: PyTree) -> Array` that returns the array $f(t)$.
        args: Other positional arguments passed to the function $f$.

    Returns:
        _(time-array object)_ Callable object returning $O(t)$ for any time $t$.
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # Pass `f` through `jax.tree_util.Partial`.
    # This is necessary:
    # (1) to make f a Pytree, and
    # (2) to avoid jitting again every time the args change.
    f = Partial(f)
    return CallableTimeArray(f, args)


def _split_shape(
    shape: tuple[int, ...], shape_1: tuple[int, ...], shape_2: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split `shape` in two shapes of the same total size as `shape_1` and `shape_2`."""
    # convert all to jnp arrays
    _shape = jnp.array(shape)
    _shape_1 = jnp.array(shape_1)
    _shape_2 = jnp.array(shape_2)

    # find total sizes
    _size = jnp.prod(_shape)
    _size_1 = jnp.prod(_shape_1)
    _size_2 = jnp.prod(_shape_2)

    # check if shape is compatible with shape_1 and shape_2
    if _size != _size_1 * _size_2:
        raise ValueError('The shape is not compatible with the shape_1 and shape_2.')

    # find where to split shape
    cumprod = jnp.cumprod(jnp.concatenate([jnp.array([1]), _shape]))
    idx = jnp.where(cumprod == _size_1)[0][-1]
    return (shape[:idx], shape[idx:])


class TimeArray(eqx.Module):
    r"""Base class for time-dependent arrays.

    A time-array is a callable object that returns a JAX array for any time $t$. It is
    used to define time-dependent operators for dynamiqs solvers.

    Attributes:
        dtype _(numpy.dtype)_: Data type.
        shape _(tuple of int)_: Shape.
        mT _(TimeArray)_: Returns the time-array transposed over its last two
            dimensions.
        ndim _(int)_: Number of dimensions.

    Notes:
        Time-arrays support elementary operations:

        - negation (`__neg__`),
        - left-and-right element-wise addition/subtraction with other arrays or
            time-arrays (`__add__`, `__radd__`, `__sub__`, `__rsub__`),
        - left-and-right element-wise multiplication with other arrays (`__mul__`,
            `__rmul__`).
    """

    # Subclasses should implement:
    # - the properties: dtype, shape, mT
    # - the methods: __call__, reshape, conj, __neg__, __mul__, __add__

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Array`, `ConstantTimeArray` and the subclass type itself.

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
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
    def ndim(self) -> int:
        return len(self.shape)

    @abstractmethod
    def reshape(self, *args: int) -> TimeArray:
        """Returns a time-array containing the same data with a new shape.

        Args:
            *args: New shape.

        Returns:
            New time-array object with the given shape.
        """

    @abstractmethod
    def conj(self) -> TimeArray:
        """Returns the element-wise complex conjugate of the time-array.

        Returns:
            New time-array object with element-wise complex conjuguated values.
        """

    @abstractmethod
    def __call__(self, t: ScalarLike) -> Array:
        """Returns the time-array evaluated at a given time.

        Args:
            t: Time at which to evaluate the array.

        Returns:
            Array evaluated at time $t$.
        """

    @abstractmethod
    def __neg__(self) -> TimeArray:
        pass

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

    def __str__(self) -> str:
        return self.__repr__()


class ConstantTimeArray(TimeArray):
    array: Array

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def mT(self) -> TimeArray:
        return ConstantTimeArray(self.array.mT)

    def __call__(self, t: ScalarLike) -> Array:  # noqa: ARG002
        return self.array

    def reshape(self, *args: int) -> TimeArray:
        return ConstantTimeArray(self.array.reshape(*args))

    def conj(self) -> TimeArray:
        return ConstantTimeArray(self.array.conj())

    def __neg__(self) -> TimeArray:
        return ConstantTimeArray(-self.array)

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
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self.values.shape[:-1], *self.array.shape)

    @property
    def mT(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values, self.array.mT)

    def __call__(self, t: float) -> Array:
        def _zero(_: float) -> Array:
            return jnp.zeros_like(self.values[..., 0])  # (...)

        def _pwc(t: float) -> Array:
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

        value = lax.cond(
            jnp.logical_or(t < self.times[0], t >= self.times[-1]), _zero, _pwc, t
        )

        return value.reshape(*value.shape, 1, 1) * self.array

    def reshape(self, *new_shape: int) -> TimeArray:
        new_values_shape, new_array_shape = _split_shape(
            new_shape, self.values.shape[:-1], self.array.shape
        )

        return PWCTimeArray(
            self.times,
            self.values.reshape(*new_values_shape, self.values.shape[-1]),
            self.array.reshape(*new_array_shape),
        )

    def conj(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values.conj(), self.array.conj())

    def __neg__(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values, -self.array)

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
    f: callable[[float, ...], Array]  # (...)
    array: Array  # (n, n)
    args: tuple[PyTree]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        f_shape = jax.eval_shape(self._call_f, 0.0).shape
        return (*f_shape, *self.array.shape)

    @property
    def mT(self) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array.mT, self.args)

    def _call_f(self, t: float) -> Array:
        args = expand_as_broadcastable(self.args)
        return self.f(t, *args)

    def __call__(self, t: float) -> Array:
        values = self._call_f(t)
        return values.reshape(*values.shape, 1, 1) * self.array

    def reshape(self, *new_shape: int) -> TimeArray:
        f_shape = jax.eval_shape(self._call_f, 0.0).shape
        new_f_shape, new_array_shape = _split_shape(
            new_shape, f_shape, self.array.shape
        )
        f = Partial(lambda t, *args: self.f(t, *args).reshape(*new_f_shape))
        return ModulatedTimeArray(f, self.array.reshape(*new_array_shape), self.args)

    def conj(self) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).conj())
        return ModulatedTimeArray(f, self.array.conj(), self.args)

    def __neg__(self) -> TimeArray:
        return ModulatedTimeArray(self.f, -self.array, self.args)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array * y, self.args)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([self, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([self, other])
        else:
            return NotImplemented


class CallableTimeArray(TimeArray):
    f: callable[[float, ...], Array]
    args: tuple[PyTree]

    @property
    def dtype(self) -> np.dtype:
        return jax.eval_shape(self._call_f, 0.0).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jax.eval_shape(self._call_f, 0.0).shape

    @property
    def mT(self) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).mT)
        return CallableTimeArray(f, self.args)

    def _call_f(self, t: float) -> Array:
        args = expand_as_broadcastable(self.args)
        return self.f(t, *args)

    def __call__(self, t: float) -> Array:
        return self._call_f(t)

    def reshape(self, *new_shape: int) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).reshape(*new_shape))
        return CallableTimeArray(f, self.args)

    def conj(self) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).conj())
        return CallableTimeArray(f, self.args)

    def __neg__(self) -> TimeArray:
        f = Partial(lambda t, *args: -self.f(t, *args))
        return CallableTimeArray(f, self.args)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args) * y)
        return CallableTimeArray(f, self.args)

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
    def dtype(self) -> np.dtype:
        return self.timearrays[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(*[tarray.shape for tarray in self.timearrays])

    @property
    def mT(self) -> TimeArray:
        return SummedTimeArray([tarray.mT for tarray in self.timearrays])

    def __call__(self, t: float) -> Array:
        return jax.tree_util.tree_reduce(
            jnp.add, [tarray(t) for tarray in self.timearrays]
        )

    def reshape(self, *new_shape: int) -> TimeArray:
        return SummedTimeArray(
            [tarray.reshape(*new_shape) for tarray in self.timearrays]
        )

    def conj(self) -> TimeArray:
        return SummedTimeArray([tarray.conj() for tarray in self.timearrays])

    def __neg__(self) -> TimeArray:
        return SummedTimeArray([-tarray for tarray in self.timearrays])

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return SummedTimeArray([tarray * y for tarray in self.timearrays])

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(other, get_args(ArrayLike)):
            other = ConstantTimeArray(jnp.asarray(other, dtype=cdtype()))
            return SummedTimeArray([*self.timearrays, other])
        elif isinstance(other, TimeArray):
            return SummedTimeArray([*self.timearrays, other])
        else:
            return NotImplemented
