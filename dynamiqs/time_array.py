from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import Tuple, get_args

from jax import Array
from jax import numpy as jnp

from .utils.utils import dag
from ._utils import check_time_array, obj_type_str, type_str
from .utils.array_types import ArrayLike, Number, dtype_complex_to_real, get_cdtype

__all__ = ['totime']


def totime(
    x: (
        ArrayLike
        | callable[[float], Array]
        | tuple[ArrayLike, ArrayLike, ArrayLike]
        | tuple[callable[[float], Array], ArrayLike]
    ),
    *,
    dtype: jnp.dtype | None = None,
) -> TimeArray:
    dtype = dtype or get_cdtype(dtype)  # assume complex by default

    # PWC time array
    if isinstance(x, tuple) and len(x) == 3:
        return _factory_pwc(x, dtype=dtype)
    # modulated time array
    if isinstance(x, tuple) and len(x) == 2:
        return _factory_modulated(x, dtype=dtype)
    # constant time array
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype)
    # callable time array
    elif callable(x):
        return _factory_callable(x, dtype=dtype)
    else:
        raise TypeError(
            'For time-dependent arrays, argument `x` must be one of 4 types: (1)'
            ' ArrayLike; (2) 2-tuple with type (function, ArrayLike) where function'
            ' has signature (t: float) -> Array; (3) 3-tuple with type (ArrayLike,'
            ' ArrayLike, ArrayLike); (4) function with signature (t: float) -> Array.'
            f' The provided `x` has type {obj_type_str(x)}.'
        )


def _factory_constant(x: ArrayLike, *, dtype: jnp.dtype) -> ConstantTimeArray:
    x = jnp.asarray(x, dtype=dtype)
    return ConstantTimeArray(x)


def _factory_callable(
    x: callable[[float], Array], *, dtype: jnp.dtype
) -> CallableTimeArray:
    f0 = x(0.0)

    # check type, dtype and device match
    if not isinstance(f0, Array):
        raise TypeError(
            f'The time-dependent operator must be a {type_str(Array)}, but has'
            f' type {obj_type_str(f0)}. The provided function must return an array,'
            ' to avoid costly type conversion at each time solver step.'
        )
    elif f0.dtype != dtype:
        raise TypeError(
            f'The time-dependent operator must have dtype `{dtype}`, but has dtype'
            f' `{f0.dtype}`. The provided function must return an array with the'
            ' same `dtype` as provided to the solver, to avoid costly dtype'
            ' conversion at each solver time step.'
        )

    return CallableTimeArray(x, f0)


def _factory_pwc(
    x: tuple[ArrayLike, ArrayLike, ArrayLike],
    *,
    dtype: jnp.dtype,
) -> PWCTimeArray:
    times, values, array = x

    # get real-valued dtype
    if dtype in (jnp.complex64, jnp.complex128):
        rdtype = dtype_complex_to_real(dtype)
    else:
        rdtype = dtype

    # times
    times = jnp.asarray(times, dtype=rdtype)
    check_time_array(times, arg_name='times')

    # values
    values = jnp.asarray(values, dtype=dtype)
    if values.shape[0] != len(times) - 1:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `values` must'
            ' have shape `(len(times)-1, ...)`, but has shape'
            f' {tuple(values.shape)}.'
        )

    # array
    array = jnp.asarray(array, dtype=dtype)
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a PWC array `(times, values, array)`, argument `array` must be'
            f' a square matrix, but has shape {tuple(array.shape)}.'
        )

    factors = [_PWCFactor(times, values)]
    arrays = array.unsqueeze(0)  # (1, n, n)
    return PWCTimeArray(factors, arrays)


def _factory_modulated(
    x: tuple[callable[[float], Array], Array],
    *,
    dtype: jnp.dtype,
) -> ModulatedTimeArray:
    f, array = x

    # get real-valued dtype
    if dtype in (jnp.complex64, jnp.complex128):
        rdtype = dtype_complex_to_real(dtype)
    else:
        rdtype = dtype

    # check f
    if not callable(f):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' be a function, but has type {obj_type_str(f)}.'
        )
    f0 = f(0.0)
    if not isinstance(f0, Array):
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `f` must'
            f' return an array, but returns type {obj_type_str(f0)}.'
        )
    if f0.dtype not in [dtype, rdtype]:
        dtypes = f'`{dtype}`' if dtype == rdtype else f'`{dtype}` or `{rdtype}`'
        raise TypeError(
            'For a modulated time array, the array returned by the function must'
            f' have dtype `{dtypes}`, but has dtype `{f0.dtype}`. This is necessary'
            ' to avoid costly dtype conversion at each solver time step.'
        )

    # array
    array = jnp.asarray(array, dtype=dtype)
    if array.ndim != 2 or array.shape[-1] != array.shape[-2]:
        raise TypeError(
            'For a modulated time array `(f, array)`, argument `array` must'
            f' be a square matrix, but has shape {tuple(array.shape)}.'
        )

    factors = [_ModulatedFactor(f, f0)]
    arrays = array.unsqueeze(0)  # (1, n, n)
    return ModulatedTimeArray(factors, arrays)


class TimeArray:
    # Subclasses should implement:
    # - the properties: dtype, device, shape
    # - the methods: __call__, reshape, adjoint, __neg__, __mul__, __add__

    # Special care should be taken when implementing `__call__` for caching to work
    # properly. The `@cache` decorator checks the array `__hash__`, which is
    # implemented as its address in memory. Thus, when two consecutive calls to a
    # `TimeArray` should return an array with the same values, these two arrays must
    # not only be equal, they should be the same object in memory.

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Array`, `ConstantTimeArray` and the subclass type itself.

    @abstractproperty
    def dtype(self) -> jnp.dtype:
        """Data type."""
        pass

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        """Shape."""
        pass

    @abstractmethod
    def __call__(self, t: float) -> Array:
        """Evaluate at a given time"""
        pass

    @abstractmethod
    def reshape(self, *shape: int) -> TimeArray:
        """Returns a new array with the same data but of a different shape."""
        pass

    @abstractmethod
    def repeat(self, n: int, axis: int) -> TimeArray:
        """Returns a new array with the same data but repeated `n` times along `axis`."""
        pass

    @abstractmethod
    def adjoint(self) -> TimeArray:
        pass

    @property
    def mH(self) -> TimeArray:
        return self.adjoint()

    @abstractmethod
    def __neg__(self) -> TimeArray:
        pass

    @abstractmethod
    def __mul__(self, other: Number | Array) -> TimeArray:
        pass

    def __rmul__(self, other: Number | Array) -> TimeArray:
        return self * other

    @abstractmethod
    def __add__(self, other: Array | TimeArray) -> TimeArray:
        pass

    def __radd__(self, other: Array | TimeArray) -> TimeArray:
        return self + other

    def __sub__(self, other: Array | TimeArray) -> TimeArray:
        return self + (-other)

    def __rsub__(self, other: Array | TimeArray) -> TimeArray:
        return other + (-self)

    def __repr__(self) -> str:
        return str(type(self).__name__)

    def __str__(self) -> str:
        return self.__repr__()

    def size(self, dim: int) -> int:
        """Size along a given dimension."""
        return self.shape[dim]

    def dim(self) -> int:
        """Get the number of dimensions."""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.dim()


class ConstantTimeArray(TimeArray):
    def __init__(self, array: Array):
        self.array = array

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    def __call__(self, t: float) -> Array:
        return self.array

    def reshape(self, *shape: int) -> TimeArray:
        return ConstantTimeArray(self.array.reshape(*shape))

    def repeat(self, n: int, axis: int) -> TimeArray:
        return ConstantTimeArray(self.array.repeat(n, axis))

    def adjoint(self) -> TimeArray:
        return ConstantTimeArray(dag(self.array))

    def __neg__(self) -> TimeArray:
        return ConstantTimeArray(-self.array)

    def __mul__(self, other: Number | Array) -> TimeArray:
        return ConstantTimeArray(self.array * other)

    def __add__(self, other: Array | TimeArray) -> TimeArray:
        if isinstance(other, Array):
            return ConstantTimeArray(self.array + other)
        elif isinstance(other, ConstantTimeArray):
            return self + other.array
        else:
            return NotImplemented


class CallableTimeArray(TimeArray):
    def __init__(self, f: callable[[float], Array], f0: Array):
        # f0 carries all the transformation on the shape
        self.f = f
        self.f0 = f0

    @property
    def dtype(self) -> jnp.dtype:
        return self.f0.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.f0.shape

    def __call__(self, t: float) -> Array:
        # cached if called twice with the same time, otherwise we recompute `f(t)`
        return self.f(t).reshape(self.shape)

    def reshape(self, *shape: int) -> TimeArray:
        f = self.f
        f0 = self.f0.reshape(*shape)
        return CallableTimeArray(f, f0)

    def repeat(self, n: int, axis: int) -> TimeArray:
        f = lambda t: self.f(t).repeat(n, axis)
        f0 = self.f0.repeat(n, axis)
        return CallableTimeArray(f, f0)

    @abstractmethod
    def adjoint(self) -> TimeArray:
        f = lambda t: self.f(t).conjugate().T
        f0 = self.f0.conjugate().T
        return CallableTimeArray(f, f0)

    def __neg__(self) -> TimeArray:
        f = lambda t: -self.f(t)
        f0 = -self.f0
        return CallableTimeArray(f, f0)

    def __mul__(self, other: Number | Array) -> TimeArray:
        f = lambda t: self.f(t) * other
        f0 = self.f0 * other
        return CallableTimeArray(f, f0)

    def __add__(self, other: Array | TimeArray) -> TimeArray:
        if isinstance(other, Array):
            f = lambda t: self.f(t) + other
            f0 = self.f0 + other
            return CallableTimeArray(f, f0)
        elif isinstance(other, ConstantTimeArray):
            return self + other.array
        elif isinstance(other, CallableTimeArray):
            f = lambda t: self.f(t) + other.f(t)
            f0 = self.f0 + other.f0
            return CallableTimeArray(f, f0)
        else:
            return NotImplemented


class _PWCFactor:
    # Defined by a tuple of 2 arrays (times, values), where
    # - times: (nv+1) are the time points between which the PWC array take constant
    #          values, where nv is the number of time intervals
    # - values: (..., nv) are the constant values for each time interval, where
    #           (...) is an arbitrary batching size

    def __init__(self, times: Array, values: Array):
        self.times = times  # (nv+1)
        self.values = values  # (..., nv)
        self.nv = self.values.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        print(self.values.shape)
        return self.values.shape[:-1]  # (...)

    def conj(self) -> _PWCFactor:
        return _PWCFactor(self.times, self.values.conj())

    def __call__(self, t: float) -> Array:
        if t < self.times[0] or t >= self.times[-1]:
            return jnp.zeros_like(self.values[..., 0])  # (...)
        else:
            # find the index $k$ such that $t \in [t_k, t_{k+1})$
            idx = jnp.searchsorted(self.times, t, side='right') - 1
            return self.values[..., idx]  # (...)

    def reshape(self, *shape: int) -> _PWCFactor:
        return _PWCFactor(self.times, self.values.reshape(*shape, self.nv))

    def repeat(self, n: int, axis: int) -> _PWCFactor:
        return _PWCFactor(self.times, self.values.repeat(n, axis))


class PWCTimeArray(TimeArray):
    # Arbitrary sum of arrays with PWC factors.

    def __init__(
        self, factors: list[_PWCFactor], arrays: Array, static: Array | None = None
    ):
        # factors must be non-empty
        self.factors = factors  # list of length (nf)
        self.arrays = arrays  # (nf, n, n)
        self.n = arrays.shape[-1]
        self.static = jnp.zeros_like(self.arrays[0]) if static is None else static

        # merge all times
        self.times = jnp.unique(jnp.concatenate([x.times for x in self.factors]))

    @property
    def dtype(self) -> jnp.dtype:
        return self.arrays.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return *self.factors[0].shape, self.n, self.n  # (..., n, n)

    def _call(self, idx: int) -> Array:
        # cache on the index in self.times

        if idx < 0 or idx >= len(self.times) - 1:
            static = self.static.reshape(*self.shape)  # (..., n, n)
            return static  # (..., n, n)
        else:
            t = self.times[idx]
            values = jnp.stack([x(t) for x in self.factors], axis=-1)  # (..., nf)
            values = values.reshape(*values.shape, 1, 1)  # (..., nf, n, n)
            return (values * self.arrays).sum(-3) + self.static  # (..., n, n)

    def __call__(self, t: float) -> Array:
        # find the index $k$ such that $t \in [t_k, t_{k+1})$, `idx = -1` if
        # `t < times[0]` and `idx = len(times) - 1` if `t >= times[-1]`
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        return self._call(idx.item())

    def reshape(self, *shape: int) -> TimeArray:
        # shape: (..., n, n)
        # TODO: @pierreguilmin : I think this implementation is incorrect and we want to reshape
        # self.arrays instead of factors, but I'm not sure
        factors = [x.reshape(*shape[:-2]) for x in self.factors]
        return PWCTimeArray(factors, self.arrays, static=self.static)

    def repeat(self, n: int, axis: int) -> TimeArray:
        # TODO: this doesn't work
        arrays = self.arrays.repeat(n, axis)
        static = self.static.repeat(n, axis)
        return PWCTimeArray(self.factors, arrays, static=static)

    def adjoint(self) -> TimeArray:
        factors = [x.conj() for x in self.factors]
        return PWCTimeArray(factors, dag(self.arrays), static=dag(self.static))

    def __neg__(self) -> TimeArray:
        return PWCTimeArray(self.factors, -self.arrays, static=-self.static)

    def __mul__(self, other: Number | Array) -> TimeArray:
        return PWCTimeArray(
            self.factors, self.arrays * other, static=self.static * other
        )

    def __add__(self, other: Array | TimeArray) -> TimeArray:
        if isinstance(other, Array):
            static = self.static + other
            return PWCTimeArray(self.factors, self.arrays, static=static)
        elif isinstance(other, ConstantTimeArray):
            return self + other.array
        elif isinstance(other, PWCTimeArray):
            factors = self.factors + other.factors  # list of length (nf1 + nf2)
            arrays = jnp.concatenate((self.arrays, other.arrays))  # (nf1 + nf2, n, n)
            static = self.static + other.static  # (n, n)
            return PWCTimeArray(factors, arrays, static=static)
        else:
            return NotImplemented


class _ModulatedFactor:
    # Defined by two objects (f, f0), where
    # - f is a callable that takes a time and returns an array of shape (...)
    # - f0 is the array of shape (...) returned by f(0.0)
    # f0 holds information about the shape of the array returned by f(t).

    def __init__(self, f: callable[[float], Array], f0: Array):
        self.f = f  # (float) -> (...)
        self.f0 = f0  # (...)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.f0.shape

    def conj(self) -> _ModulatedFactor:
        f = lambda t: self.f(t).conj()
        f0 = self.f0.conj()
        return _ModulatedFactor(f, f0)

    def __call__(self, t: float) -> Array:
        return self.f(t).reshape(self.shape)

    def reshape(self, *shape: int) -> _ModulatedFactor:
        f = self.f
        f0 = self.f0.reshape(*shape)
        return _ModulatedFactor(f, f0)

    def repeat(self, n: int, axis: int) -> _ModulatedFactor:
        f = self.f
        f0 = self.f0.repeat(n, axis)
        return _ModulatedFactor(f, f0)


class ModulatedTimeArray(TimeArray):
    # Sum of arrays with callable factors.

    def __init__(
        self,
        factors: list[_ModulatedFactor],
        arrays: Array,
        static: Array | None = None,
    ):
        # factors must be non-empty
        self.factors = factors  # list of length (nf)
        self.arrays = arrays  # (nf, n, n)
        self.n = arrays.shape[-1]
        self.static = jnp.zeros_like(self.arrays[0]) if static is None else static

    @property
    def dtype(self) -> jnp.dtype:
        return self.arrays.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return jnp.Size((*self.factors[0].shape, self.n, self.n))  # (..., n, n)

    def __call__(self, t: float) -> Array:
        values = jnp.stack([x(t) for x in self.factors], axis=-1)  # (..., nf)
        values = values.reshape(*values.shape, 1, 1)  # (..., nf, n, n)
        print(values.shape, self.arrays.shape, self.static.shape)
        return (values * self.arrays).sum(-3) + self.static  # (..., n, n)

    def reshape(self, *shape: int) -> TimeArray:
        # shape: (..., n, n)
        factors = [x.reshape(*shape[:-2]) for x in self.factors]
        return ModulatedTimeArray(factors, self.arrays, static=self.static)

    def repeat(self, n: int, axis: int) -> TimeArray:
        arrays = self.arrays.repeat(n, axis + 1)
        static = self.static.repeat(n, axis)
        return ModulatedTimeArray(self.factors, arrays, static=static)

    def adjoint(self) -> TimeArray:
        factors = [x.conj() for x in self.factors]
        return ModulatedTimeArray(factors, dag(self.arrays), static=dag(self.static))

    def __neg__(self) -> TimeArray:
        return ModulatedTimeArray(self.factors, -self.arrays, static=-self.static)

    def __mul__(self, other: Number | Array) -> TimeArray:
        return ModulatedTimeArray(
            self.factors, self.arrays * other, static=self.static * other
        )

    def __add__(self, other: Array | TimeArray) -> TimeArray:
        if isinstance(other, Array):
            static = self.static + other
            return ModulatedTimeArray(self.factors, self.arrays, static=static)
        elif isinstance(other, ConstantTimeArray):
            return self + other.array
        elif isinstance(other, ModulatedTimeArray):
            factors = self.factors + other.factors  # list of length (nf1 + nf2)
            arrays = jnp.concatenate((self.arrays, other.arrays))  # (nf1 + nf2, n, n)
            static = self.static + other.static  # (n, n)
            return ModulatedTimeArray(factors, arrays, static=static)
        else:
            return NotImplemented
