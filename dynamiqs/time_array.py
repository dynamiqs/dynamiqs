from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Union, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import Partial
from jaxtyping import PyTree, Scalar

from ._utils import check_time_array, obj_type_str, type_str
from .utils.array_types import ArrayLike, cdtype

__all__ = ['totime']

TimeArrayLike = Union[
    ArrayLike,
    Callable[[float, tuple[PyTree]], Array],
    tuple[ArrayLike, ArrayLike, ArrayLike],
    tuple[Callable[[float, tuple[PyTree]], Array], ArrayLike],
    'TimeArray',
]


def totime(x: TimeArrayLike, *, args: tuple[PyTree] = ()) -> TimeArray:
    r"""Instantiate a time-dependent array of type `TimeArray`.

    There are 4 ways to define a time-dependent array in dynamiqs.

    **1/ Constant time array** – A constant array of the form $A(t) = A_0$. It is
    initialized with `x = A0` as an array-like object:

    - **A0** _(array-like)_ – The constant array $A_0$, of shape _(..., n, n)_.

    **2/ Modulated time array** – A modulated time array of the form $A(t) = f(t) A_0$.
    It is initialized with `x = (f, A0)`, where:

    - **f** _(function)_ – A function with signature `(t: float, *args: PyTree) ->
    Array` that returns the modulating factor $f(t)$ of shape _(...,)_.
    - **A0** _(array-like)_ – The constant array $A_0$, of shape _(n, n)_.

    **3/ PWC time array** – A piecewise-constant (PWC) time array of the form $A(t) =
    A_i$ for $t \in [t_i, t_{i+1})$. It is initialized with `x = (times, values, array)`, where:

    - **times** _(array-like)_ – The time points $t_i$ between which the PWC factor
    takes constant values, of shape _(nv+1,)_ where _nv_ is the number of time
    intervals.
    - **values** _(array-like)_ – The constant values for each time interval, of shape
    _(..., nv)_.
    - **array** _(array-like)_ – The constant array $A_i$, of shape _(n, n)_.

    **4/ Callable time array** – A time array defined by a callable function, of
    generic form $A(t) = f(t)$. It is initialized with `x = f` as:

    - **f** _(function)_ – A function with signature `(t: float, *args: PyTree) ->
    Array` with shape _(..., n, n)_.

    Note: TimeArrays
        A `TimeArray` object has several attributes and methods, including:

        - **self.dtype** – Returns the data type of the array.
        - **self.shape** – Returns the shape of the array.
        - **self.mT** – Returns the transpose of the array.
        - **self(t: float)** – Evaluates the array at a given time.
        - **self.reshape(*args: int)** – Returns an array containing the same data with
            a new shape.
        - **self.conj()** – Returns the complex conjugate, element-wise.

        `TimeArray` objects also support the following operations:

        - **-self** – Returns the negation of the array.
        - **y * self** – Returns the product of `y` with the array, where `y` is an
            array-like broadcastable with `self`.
        - **self + other** – Returns the sum of the array and `other`, where `other` is
            an array-like object or another instance of `TimeArray`.

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


def _factory_callable(
    f: callable[[float, tuple[PyTree]], Array], *, args: tuple[PyTree]
) -> CallableTimeArray:
    f0 = f(0.0, *args)

    # check type, dtype and device match
    if not isinstance(f0, Array):
        raise TypeError(
            f'The time-dependent operator must be a {type_str(Array)}, but has'
            f' type {obj_type_str(f0)}. The provided function must return an array,'
            ' to avoid costly type conversion at each time solver step.'
        )
    elif f0.dtype != cdtype():
        raise TypeError(
            f'The time-dependent operator must have dtype `{cdtype()}`, but has dtype'
            f' `{f0.dtype}`. The provided function must return an array with the'
            ' same `dtype` as provided to the solver, to avoid costly dtype'
            ' conversion at each solver time step.'
        )

    # Pass `f` through `jax.tree_util.Partial`.
    # This is necessary:
    # (1) to make f a Pytree, and
    # (2) to avoid jitting again every time the args change.
    f_partial = Partial(lambda t: f(t, *args))
    return CallableTimeArray(f_partial, f0)


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

    factors = [_PWCFactor(times, values)]
    arrays = array[None, ...]  # (1, n, n)
    static = jnp.zeros_like(array)
    return PWCTimeArray(factors, arrays, static=static)


def _factory_modulated(
    x: tuple[callable[[float, tuple[PyTree]], Array], Array], *, args: tuple[PyTree]
) -> ModulatedTimeArray:
    f, array = x

    # check f
    if not callable(f):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' be a function, but has type {obj_type_str(f)}.'
        )
    f0 = f(0.0, *args)
    if not isinstance(f0, Array):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' return an array, but returns type {obj_type_str(f0)}.'
        )
    # todo: do we really need this?
    # if f0.dtype not in [dtype, rdtype]:
    #     dtypes = f'`{dtype}`' if dtype == rdtype else f'`{dtype}` or `{rdtype}`'
    #     raise TypeError(
    #         'For a modulated time array, the array returned by the function must'
    #         f' have dtype `{dtypes}`, but has dtype `{f0.dtype}`. This is necessary'
    #         ' to avoid costly dtype conversion at each solver time step.'
    #     )

    # array
    array = jnp.asarray(array, dtype=cdtype())
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `array` must'
            f' be a square matrix, but has shape {tuple(array.shape)}.'
        )

    factors = [_ModulatedFactor(Partial(f, *args), f0)]
    arrays = array[None, ...]  # (1, n, n)
    static = jnp.zeros_like(array)
    return ModulatedTimeArray(factors, arrays, static=static)


class SumCallable(eqx.Module):
    f1: callable[[float], Array]
    f2: callable[[float], Array]

    def __call__(self, t: float) -> Array:
        return self.f1(t) + self.f2(t)


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
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""
        pass

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
        pass

    @abstractmethod
    def reshape(self, *args: int) -> TimeArray:
        """Returns an array containing the same data with a new shape."""
        pass

    @abstractmethod
    def conj(self) -> TimeArray:
        """Return the complex conjugate, element-wise."""
        pass

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

    def __call__(self, t: Scalar) -> Array:
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
        if isinstance(y, get_args(ArrayLike)):
            return ConstantTimeArray(self.x + y)
        elif isinstance(y, ConstantTimeArray):
            return ConstantTimeArray(self.x + y.x)
        else:
            return NotImplemented


class CallableTimeArray(TimeArray):
    f: callable[[float], Array]
    f0: Array

    @property
    def dtype(self) -> np.dtype:
        return self.f0.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f0.shape

    @property
    def mT(self) -> TimeArray:
        f = lambda t: self.f(t).mT
        f0 = self.f0.mT
        return CallableTimeArray(f, f0)

    def __call__(self, t: float) -> Array:
        return self.f(t).reshape(*self.shape)

    def reshape(self, *args: int) -> TimeArray:
        f = self.f
        f0 = self.f0.reshape(*args)
        return CallableTimeArray(f, f0)

    def conj(self) -> TimeArray:
        f = lambda t: self.f(t).conj()
        f0 = self.f0.conj()
        return CallableTimeArray(f, f0)

    def __neg__(self) -> TimeArray:
        f = lambda t: -self.f(t)
        f0 = -self.f0
        return CallableTimeArray(f, f0)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        f = lambda t: self.f(t) * y
        f0 = self.f0 * y
        return CallableTimeArray(f, f0)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(y, get_args(ArrayLike)):
            f = SumCallable(self.f, ConstantTimeArray(y))
            f0 = self.f0 + y
            return CallableTimeArray(f, f0)
        elif isinstance(y, ConstantTimeArray):
            f = SumCallable(self.f, y)
            f0 = self.f0 + y.x
            return CallableTimeArray(f, f0)
        elif isinstance(y, CallableTimeArray):
            f = SumCallable(self.f, y.f)
            f0 = self.f0 + y.f0
            return CallableTimeArray(f, f0)
        else:
            return NotImplemented


class _Factor(eqx.Module):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def conj(self) -> _Factor:
        pass

    @abstractmethod
    def __call__(self, t: Scalar) -> Array:
        pass

    @abstractmethod
    def reshape(self, *args: int) -> _Factor:
        pass


class _PWCFactor(_Factor):
    # Defined by a tuple of 2 arrays (times, values), where
    # - times: (nv+1) are the time points between which the PWC factor takes constant
    #          values, where nv is the number of time intervals
    # - values: (..., nv) are the constant values for each time interval, where (...)
    #           is an arbitrary batching size
    times: Array
    values: Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]  # (...)

    def conj(self) -> _Factor:
        return _PWCFactor(self.times, self.values.conj())

    def __call__(self, t: Scalar) -> Array:
        def _zero(t: Scalar) -> Array:
            return jnp.zeros_like(self.values[..., 0])  # (...)

        def _pwc(t: Scalar) -> Array:
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

        return lax.cond(
            jnp.logical_or(t < self.times[0], t >= self.times[-1]), _zero, _pwc, t
        )

    def reshape(self, *args: int) -> _Factor:
        return _PWCFactor(self.times, self.values.reshape(*args, self.values.shape[-1]))


class _ModulatedFactor(_Factor):
    # Defined by two objects (f, f0), where
    # - f is a callable that takes a time and returns an array of shape (...)
    # - f0 is the array of shape (...) returned by f(0.0)
    # f0 holds information about the shape of the array returned by f(t).
    f: callable[[float], Array]  # (float) -> (...)
    f0: Array  # (...)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f0.shape

    def conj(self) -> _Factor:
        f = lambda t: self.f(t).conj()
        f0 = self.f0.conj()
        return _ModulatedFactor(f, f0)

    def __call__(self, t: Scalar) -> Array:
        return self.f(t).reshape(self.shape)

    def reshape(self, *args: int) -> _Factor:
        f = self.f
        f0 = self.f0.reshape(*args)
        return _ModulatedFactor(f, f0)


class FactorTimeArray(TimeArray):
    factors: list[_Factor]  # list of length (nf) - must be non-empty
    arrays: Array  # (nf, n, n)
    static: Array  # (n, n)

    @property
    def dtype(self) -> np.dtype:
        return self.arrays.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        n = self.arrays.shape[-1]
        return tuple(self.factors[0].shape, n, n)  # (..., n, n)

    @property
    def mT(self) -> TimeArray:
        return FactorTimeArray(self.factors, self.arrays.mT, static=self.static.mT)

    def __call__(self, t: float) -> Array:
        values = jnp.stack([x(t) for x in self.factors], axis=-1)  # (..., nf)
        values = values.reshape(*values.shape, 1, 1)  # (..., nf, n, n)
        return (values * self.arrays).sum(-3) + self.static  # (..., n, n)

    def reshape(self, *args: int) -> TimeArray:
        # shape: (..., n, n)
        factors = [x.reshape(*args[:-2]) for x in self.factors]
        return self.__class__(factors, self.arrays, static=self.static)

    def conj(self) -> TimeArray:
        factors = [x.conj() for x in self.factors]
        return self.__class__(factors, self.arrays.conj(), static=self.static.conj())

    def __neg__(self) -> TimeArray:
        return self.__class__(self.factors, -self.arrays, static=-self.static)

    def __mul__(self, y: ArrayLike) -> TimeArray:
        return self.__class__(self.factors, self.arrays * y, static=self.static * y)

    def __add__(self, y: ArrayLike | TimeArray) -> TimeArray:
        if isinstance(y, get_args(ArrayLike)):
            static = self.static + y
            return self.__class__(self.factors, self.arrays, static=static)
        elif isinstance(y, ConstantTimeArray):
            static = self.static + y.x
            return self.__class__(self.factors, self.arrays, static=static)
        elif isinstance(y, self.__class__):
            factors = self.factors + y.factors  # list of length (nf1 + nf2)
            arrays = jnp.concatenate((self.arrays, y.arrays))  # (nf1 + nf2, n, n)
            static = self.static + y.static  # (n, n)
            return self.__class__(factors, arrays, static=static)
        else:
            return NotImplemented


class PWCTimeArray(FactorTimeArray):
    # Arbitrary sum of arrays with PWC factors.
    times: Array

    def __init__(
        self,
        factors: list[_PWCFactor],
        arrays: Array,
        static: Array,
    ):
        super().__init__(factors, arrays, static)
        self.times = jnp.unique(jnp.concatenate([x.times for x in self.factors]))


class ModulatedTimeArray(FactorTimeArray):
    # Sum of arrays with callable factors.
    pass
