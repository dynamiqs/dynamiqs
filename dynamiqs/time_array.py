from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Union, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import Partial
from jaxtyping import PyTree, Scalar

from ._utils import check_time_array, obj_type_str
from .utils.array_types import ArrayLike, cdtype

__all__ = ['totime']

TimeArrayLike = Union[
    ArrayLike,
    Callable[[float, ...], Array],
    tuple[ArrayLike, ArrayLike, ArrayLike],
    tuple[Callable[[float, ...], Array], ArrayLike],
    'TimeArray',
]


def totime(x: TimeArrayLike, *args: PyTree) -> TimeArray:
    r"""Instantiate a time-dependent array of type `TimeArray`.

    There are 4 ways to define a time-dependent array in dynamiqs.

    **(1) Constant time array:** A constant array of the form $A(t) = A_0$. It is
    initialized with `x = A0` as an array-like object:

    - **A0** _(array-like)_ - The constant array $A_0$, of shape _(..., n, n)_.

    **(2) PWC time array:** A piecewise-constant (PWC) time array of the form $A(t) =
    A_i$ for $t \in [t_i, t_{i+1})$. It is initialized with
    `x = (times, values, array)`, where:

    - **times** _(array-like)_ - The time points $t_i$ between which the PWC factor
    takes constant values, of shape _(nv+1,)_ where _nv_ is the number of time
    intervals.
    - **values** _(array-like)_ - The constant values for each time interval, of shape
    _(..., nv)_.
    - **array** _(array-like)_ - The constant array $A_i$, of shape _(n, n)_.

    **(3) Modulated time array:** A modulated time array of the form $A(t) = f(t) A_0$.
    It is initialized with `x = (f, A0)`, where:

    - **f** _(function)_ - A function with signature `(t: float, *args: PyTree) ->
    Array` that returns the modulating factor $f(t)$ of shape _(...,)_.
    - **A0** _(array-like)_ - The constant array $A_0$, of shape _(n, n)_.

    **(4) Callable time array:** A time array defined by a callable function, of
    generic form $A(t) = f(t)$. It is initialized with `x = f` as:

    - **f** _(function)_ - A function with signature `(t: float, *args: PyTree) ->
    Array` with shape _(..., n, n)_.

    Note: TimeArrays
        A `TimeArray` object has several attributes and methods, including:

        - **self.dtype** - Returns the data type of the array.
        - **self.shape** - Returns the shape of the array.
        - **self.mT** - Returns the transpose of the array.
        - **self(t: float)** - Evaluates the array at a given time.
        - **self.reshape(*args: int)** - Returns an array containing the same data with
            a new shape.
        - **self.conj()** - Returns the complex conjugate, element-wise.

        `TimeArray` objects also support the following operations:

        - **-self** - Returns the negation of the array.
        - **y * self** - Returns the product of `y` with the array, where `y` is an
            array-like broadcastable with `self`.
        - **self + other** - Returns the sum of the array and `other`, where `other` is
            an array-like object or another instance of `TimeArray`.

    Note: Batching over callable and modulated time arrays
        To batch over callable and modulated time arrays, the function `f` must take
        its batched array as extra argument. For example, here are two correct
        implementations of the Hamiltonian $H(t) = \sigma_z + \cos(\omega t) \sigma_x$
        with batching over $\omega$:
        ```python
        import jax.numpy as jnp

        # array to batch over
        omegas = jnp.linspace(-1.0, 1.0, 20)

        # using a modulated time array
        def cx(t, omega):
            return jnp.cos(t * omega)
        H = dq.sigmaz() + dq.totime((cx, dq.sigmax()), omegas)

        # using a callable time array
        def Hx(t, omega):
            return jnp.cos(t * jnp.expand_dims(omega, (-1, -2))) * dq.sigmax()
        H = dq.sigmaz() + dq.totime(Hx, omegas)
        ```

    Args:
        x: The time-dependent array initializer.
        args: The extra arguments passed to the function for modulated and callable
            time arrays.
    """
    # already a time array
    if isinstance(x, TimeArray):
        return x
    # PWC time array
    elif isinstance(x, tuple) and len(x) == 3:
        return _factory_pwc(x)
    # modulated time array
    elif isinstance(x, tuple) and len(x) == 2:
        return _factory_modulated(x, args=args)
    # constant time array
    elif isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x)
    # callable time array
    elif callable(x):
        return _factory_callable(x, args=args)
    else:
        raise TypeError(
            'For time-dependent arrays, argument `x` must be one of 4 types: (1)'
            ' ArrayLike; (2) 2-tuple with type (function, ArrayLike) where function'
            ' has signature (t: float, *args: PyTree) -> Array; (3) 3-tuple with type'
            ' (ArrayLike, ArrayLike, ArrayLike); (4) function with signature (t:'
            ' float, *args: PyTree) -> Array. The provided `x` has type'
            f' {obj_type_str(x)}.'
        )


def _factory_constant(x: ArrayLike) -> ConstantTimeArray:
    x = jnp.asarray(x, dtype=cdtype())
    return ConstantTimeArray(x)


def _factory_pwc(x: tuple[ArrayLike, ArrayLike, ArrayLike]) -> PWCTimeArray:
    times, values, array = x

    # times
    times = jnp.asarray(times)
    check_time_array(times, arg_name='times')

    # values
    values = jnp.asarray(values, dtype=cdtype())
    if values.shape[0] != len(times) - 1:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `values` must'
            ' have shape `(len(times)-1, ...)`, but has shape'
            f' {tuple(values.shape)}.'
        )

    # array
    array = jnp.asarray(array, dtype=cdtype())
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `array` must be'
            f' a square matrix, but has shape {tuple(array.shape)}.'
        )

    return PWCTimeArray(times, values, array)


def _factory_modulated(
    x: tuple[callable[[float, ...], Array], Array], *, args: tuple[PyTree]
) -> ModulatedTimeArray:
    f, array = x

    # check f is callable
    if not callable(f):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' be a function, but has type {obj_type_str(f)}.'
        )

    # array
    array = jnp.asarray(array, dtype=cdtype())
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `array` must'
            f' be a square matrix, but has shape {tuple(array.shape)}.'
        )

    # Pass `f` through `jax.tree_util.Partial`.
    # This is necessary:
    # (1) to make f a Pytree, and
    # (2) to avoid jitting again every time the args change.
    f = Partial(f)

    return ModulatedTimeArray(f, array, args)


def _factory_callable(
    f: callable[[float, ...], Array], *, args: tuple[PyTree]
) -> CallableTimeArray:
    # check f is callable
    if not callable(f):
        raise TypeError(
            'For a callable time array, argument `f` must be a function, but has type'
            f' {obj_type_str(f)}.'
        )

    # Pass `f` through `jax.tree_util.Partial`.
    # This is necessary:
    # (1) to make f a Pytree, and
    # (2) to avoid jitting again every time the args change.
    f = Partial(f)
    return CallableTimeArray(f, args)


class TimeArray(eqx.Module):
    # Subclasses should implement:
    # - the properties: dtype, shape, mT
    # - the methods: __call__, reshape, conj, __neg__, __mul__, __add__

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Array`, `ConstantTimeArray` and the subclass type itself.

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type (numpy.dtype) of the array."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""

    @property
    @abstractmethod
    def mT(self) -> TimeArray:
        """Transposes the last two dimensions of x."""

    @property
    def ndim(self) -> int:
        """The number of dimensions in the array."""
        return len(self.shape)

    @abstractmethod
    def __call__(self, t: Scalar) -> Array:
        """Evaluate at a given time."""

    @abstractmethod
    def reshape(self, *args: int) -> TimeArray:
        """Returns an array containing the same data with a new shape."""

    @abstractmethod
    def conj(self) -> TimeArray:
        """Return the complex conjugate, element-wise."""

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
    x: Array

    @property
    def dtype(self) -> np.dtype:
        return self.x.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.shape

    @property
    def mT(self) -> TimeArray:
        return ConstantTimeArray(self.x.mT)

    def __call__(self, t: Scalar) -> Array:  # noqa: ARG002
        return self.x

    def reshape(self, *args: int) -> TimeArray:
        return ConstantTimeArray(self.x.reshape(*args))

    def conj(self) -> TimeArray:
        return ConstantTimeArray(self.x.conj())

    def __neg__(self) -> TimeArray:
        return ConstantTimeArray(-self.x)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ConstantTimeArray(self.x * y)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return NotImplemented


class PWCTimeArray(TimeArray):
    # `array`` is made a static field such that `vmap` knows to not batch over it.
    # However, this also implies that the function is re-jitted every time `array`
    # changes. TODO: find a better way to handle this.
    times: Array  # (nv+1,)
    values: Array  # (..., nv)
    array: Array = eqx.field(static=True)  # (n, n)

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

        return value.reshape(*value, 1, 1) * self.array

    def reshape(self, *new_shape: int) -> TimeArray:
        # there may be a better way to handle this
        if new_shape[0] != self.shape[0]:
            raise ValueError(
                'The first dimension of the new shape must match the batching dimension'
                f' of the time array, but got {new_shape[0]} and {self.shape[0]}.'
            )
        return PWCTimeArray(self.times, self.values, self.array.reshape(*new_shape[1:]))

    def conj(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values.conj(), self.array.conj())

    def __neg__(self) -> TimeArray:
        return PWCTimeArray(self.times, self.values, -self.array)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return PWCTimeArray(self.times, self.values, self.array * y)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        return NotImplemented


class ModulatedTimeArray(TimeArray):
    # `array`` is made a static field such that `vmap` knows to not batch over it.
    # However, this also implies that the function is re-jitted every time `array`
    # changes. TODO: find a better way to handle this.
    f: callable[[float, ...], Array]  # (...,)
    array: Array = eqx.field(static=True)  # (n, n)
    args: tuple[PyTree]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        f_shape = jax.eval_shape(self.f, 0.0, *self.args).shape
        return (*f_shape, *self.array.shape)

    @property
    def mT(self) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array.mT, self.args)

    def __call__(self, t: float) -> Array:
        values = self.f(t, *self.args)
        return values.reshape(*values.shape, 1, 1) * self.array

    def reshape(self, *new_shape: int) -> TimeArray:
        # there may be a better way to handle this
        if new_shape[0] != self.shape[0]:
            raise ValueError(
                'The first dimension of the new shape must match the batching dimension'
                f' of the time array, but got {new_shape[0]} and {self.shape[0]}.'
            )
        return ModulatedTimeArray(self.f, self.array.reshape(*new_shape[1:]), self.args)

    def conj(self) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).conj())
        return ModulatedTimeArray(f, self.array.conj(), self.args)

    def __neg__(self) -> TimeArray:
        return ModulatedTimeArray(self.f, -self.array, self.args)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return ModulatedTimeArray(self.f, self.array * y, self.args)

    def __add__(self, other: ArrayLike | TimeArray) -> TimeArray:
        return NotImplemented


class CallableTimeArray(TimeArray):
    f: callable[[float, ...], Array]
    args: tuple[PyTree]

    @property
    def dtype(self) -> np.dtype:
        return jax.eval_shape(self.f, 0.0, *self.args).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0, *self.args).shape

    @property
    def mT(self) -> TimeArray:
        f = Partial(lambda t, *args: self.f(t, *args).mT)
        return CallableTimeArray(f, self.args)

    def __call__(self, t: float) -> Array:
        return self.f(t, *self.args)

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

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        return NotImplemented
