from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from math import prod
from typing import Any, Union, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .layout import Layout

__all__ = ['QArray', 'QArrayLike', 'isqarraylike']


def isqarraylike(x: Any) -> bool:
    if isinstance(x, get_args(_QArrayLike)):
        return True
    elif isinstance(x, Sequence):
        return all(isqarraylike(sub_x) for sub_x in x)
    return False


def _to_jax(x: QArrayLike) -> Array:
    if isinstance(x, QArray):
        return x.to_jax()
    elif isinstance(x, Qobj):
        return jnp.asarray(x.full())
    elif isinstance(x, Sequence):
        return jnp.asarray([_to_jax(sub_x) for sub_x in x])
    else:
        return jnp.asarray(x)


def _to_numpy(x: QArrayLike) -> np.ndarray:
    if isinstance(x, QArray):
        return x.to_numpy()
    elif isinstance(x, Qobj):
        return np.asarray(x.full())
    elif isinstance(x, Sequence):
        return np.asarray([_to_numpy(sub_x) for sub_x in x])
    else:
        return np.asarray(x)


def _dims_to_qutip(dims: tuple[int, ...], shape: tuple[int, ...]) -> list[list[int]]:
    dims = list(dims)
    if shape[-1] == 1:  # [[3], [1]] or [[3, 4], [1, 1]]
        dims = [dims, [1] * len(dims)]
    elif shape[-2] == 1:  # [[1], [3]] or [[1, 1], [3, 4]]
        dims = [[1] * len(dims), dims]
    elif shape[-1] == shape[-2]:  # [[3], [3]] or [[3, 4], [3, 4]]
        dims = [dims, dims]
    return dims


def _dims_from_qutip(dims: list[list[int]]) -> tuple[int, ...]:
    dims = np.max(dims, axis=0)
    return tuple(dims.tolist())


class QArray(eqx.Module):
    r"""Dynamiqs custom array to represent quantum objects.

    A qarray is a wrapper around the data structure representing a quantum object (a
    ket, a density matrix, an operator, a superoperator, etc.) that offers convenience
    methods.

    There are two types of qarrays:

    - `DenseQArray`: wrapper around JAX arrays, dense representation of the array.
    - `SparseDIAQArray`: Dynamiqs sparse diagonal format, storing only the non-zero
        diagonals.

    Use the constructor [`dq.asqarray()`][dynamiqs.asqarray] to build a qarray.

    Attributes:
        dtype _(numpy dtype)_: Data type.
        shape _(tuple of ints)_: Array shape.
        ndim _(int)_: Number of dimensions of the array.
        layout _(Layout)_: Data layout, either `dq.dense` or `dq.dia`.
        dims _(tuple of ints)_: Hilbert space dimension of each subsystem.
        mT _(QArray)_: Returns the qarray transposed over its last two dimensions.

    Note: Arithmetic operation support
        Qarrays support elementary operations, such as element-wise
        addition/subtraction, element-wise multiplication and matrix multiplications
        with other qarray-like objects.

    Note-: Shortcuts methods to use quantum utilities
        Many functions of the library can be called directly on a qarray rather than
        through the functional API. For example, you can use `x.dag()` instead of
        `dq.isdag(x)`. Here is the complete list of these shortcuts:

        | QArray method    | Corresponding function call                          |
        |------------------|------------------------------------------------------|
        | `x.dag()`        | [`dq.dag(x)`][dynamiqs.dag]                          |
        | `x.powm()`       | [`dq.powm(x)`][dynamiqs.powm]                        |
        | `x.expm()`       | [`dq.expm(x)`][dynamiqs.expm]                        |
        | `x.cosm()`       | [`dq.cosm(x)`][dynamiqs.cosm]                        |
        | `x.sinm()`       | [`dq.sinm(x)`][dynamiqs.sinm]                        |
        | `x.trace()`      | [`dq.trace(x)`][dynamiqs.trace]                      |
        | `x.ptrace(keep)` | [`dq.ptrace(x, keep, dims=x.dims)`][dynamiqs.ptrace] |
        | `x.norm()`       | [`dq.norm(x)`][dynamiqs.norm]                        |
        | `x.unit()`       | [`dq.unit(x)`][dynamiqs.unit]                        |
        | `x.isket()`      | [`dq.isket(x)`][dynamiqs.isket]                      |
        | `x.isbra()`      | [`dq.isbra(x)`][dynamiqs.isbra]                      |
        | `x.isdm()`       | [`dq.isdm(x)`][dynamiqs.isdm]                        |
        | `x.isop()`       | [`dq.isop(x)`][dynamiqs.isop]                        |
        | `x.isherm()`     | [`dq.isherm(x)`][dynamiqs.isherm]                    |
        | `x.toket()`      | [`dq.toket(x)`][dynamiqs.toket]                      |
        | `x.tobra()`      | [`dq.tobra(x)`][dynamiqs.tobra]                      |
        | `x.todm()`       | [`dq.todm(x)`][dynamiqs.todm]                        |
        | `x.proj()`       | [`dq.proj(x)`][dynamiqs.proj]                        |
        | `x.to_qutip()`   | [`dq.to_qutip(x, dims=x.dims)`][dynamiqs.to_qutip]   |
        | `x.to_jax()`     | [`dq.to_jax(x)`][dynamiqs.to_jax]                    |
        | `x.to_numpy()`   | [`dq.to_numpy(x)`][dynamiqs.to_numpy]                |
    """

    # Subclasses should implement:
    # - the properties: dtype, layout, shape, mT
    # - the methods:
    #   - QArray methods: conj, dag, reshape, broadcast_to, ptrace, powm, expm,
    #                     _abs
    #   - returning a JAX array or other: norm, trace, sum, squeeze, _eigh, _eigvals,
    #                                     _eigvalsh, devices, isherm
    #   - conversion methods: to_qutip, to_jax, __array__
    #   - special methods: __mul__, __truediv__, __add__, __matmul__, __rmatmul__,
    #                         __and__, _pow, __getitem__

    # TODO: Setting dims as static for now. Otherwise, I believe it is upgraded to a
    # complex dtype during the computation, which raises an error on diffrax side.
    dims: tuple[int, ...] = eqx.field(static=True)

    def __check_init__(self):
        # === ensure dims is a tuple of ints
        if not isinstance(self.dims, tuple) or not all(
            isinstance(d, int) for d in self.dims
        ):
            raise TypeError(
                f'Argument `dims` must be a tuple of ints, but is {self.dims}.'
            )

        # === ensure dims is compatible with the shape
        # for vectorized superoperators, we allow that the shape is the square
        # of the product of all dims
        allowed_shapes = (prod(self.dims), prod(self.dims) ** 2)
        if not (self.shape[-1] in allowed_shapes or self.shape[-2] in allowed_shapes):
            raise ValueError(
                'Argument `dims` must be compatible with the shape of the QArray, but '
                f'got dims {self.dims} and shape {self.shape}.'
            )

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        pass

    @property
    @abstractmethod
    def layout(self) -> Layout:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def mT(self) -> QArray:
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @abstractmethod
    def conj(self) -> QArray:
        """Returns the element-wise complex conjugate of the qarray.

        Returns:
            New qarray object with element-wise complex conjuguated values.
        """

    def dag(self) -> QArray:
        return self.mT.conj()

    @abstractmethod
    def reshape(self, *shape: int) -> QArray:
        """Returns a reshaped copy of a qarray.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New qarray object with the given shape.
        """

    @abstractmethod
    def broadcast_to(self, *shape: int) -> QArray:
        """Broadcasts a qarray to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New qarray object with the given shape.
        """

    @abstractmethod
    def ptrace(self, *keep: int) -> QArray:
        pass

    @abstractmethod
    def powm(self, n: int) -> QArray:
        pass

    @abstractmethod
    def expm(self, *, max_squarings: int = 16) -> QArray:
        pass

    def cosm(self) -> QArray:
        from ..utils import cosm

        return cosm(self)

    def sinm(self) -> QArray:
        from ..utils import sinm

        return sinm(self)

    def unit(self) -> QArray:
        return self / self.norm()[..., None, None]

    @abstractmethod
    def norm(self) -> Array:
        pass

    @abstractmethod
    def trace(self) -> Array:
        pass

    @abstractmethod
    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # todo
        pass

    @abstractmethod
    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # todo
        pass

    @abstractmethod
    def _eigh(self) -> tuple[Array, Array]:
        pass

    @abstractmethod
    def _eigvals(self) -> Array:
        pass

    @abstractmethod
    def _eigvalsh(self) -> Array:
        pass

    @abstractmethod
    def devices(self) -> set[Device]:
        # todo
        pass

    def isket(self) -> bool:
        from ..utils import isket

        return isket(self)

    def isbra(self) -> bool:
        from ..utils import isbra

        return isbra(self)

    def isdm(self) -> bool:
        from ..utils import isdm

        return isdm(self)

    def isop(self) -> bool:
        from ..utils import isop

        return isop(self)

    @abstractmethod
    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        pass

    def toket(self) -> QArray:
        from ..utils import toket

        return toket(self)

    def tobra(self) -> QArray:
        from ..utils import tobra

        return tobra(self)

    def todm(self) -> QArray:
        from ..utils import todm

        return todm(self)

    def proj(self) -> QArray:
        from ..utils import proj

        return proj(self)

    @abstractmethod
    def to_qutip(self) -> Qobj | list[Qobj]:
        pass

    @abstractmethod
    def to_jax(self) -> Array:
        pass

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError('len() of unsized object') from err

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        pass

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self)

    def __repr__(self) -> str:
        return (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'layout={self.layout}'
        )

    def __neg__(self) -> QArray:
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: QArrayLike) -> QArray:
        from .._utils import _is_batched_scalar

        if not _is_batched_scalar(y):
            logging.warning(
                'Using the `*` operator between two arrays performs element-wise '
                'multiplication. For matrix multiplication, use the `@` operator '
                'instead.'
            )

        if isinstance(y, QArray):
            _check_compatible_dims(self.dims, y.dims)

    def __rmul__(self, y: QArrayLike) -> QArray:
        return self * y

    @abstractmethod
    def __truediv__(self, y: QArrayLike) -> QArray:
        pass

    def __rtruediv__(self, y: QArrayLike) -> QArray:
        return self * 1 / y

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    @abstractmethod
    def __add__(self, y: QArrayLike) -> QArray:
        from .._utils import _is_batched_scalar

        if _is_batched_scalar(y):
            logging.warning(
                'Using the `+` or `-` operator between an array and a scalar performs '
                'element-wise addition or subtraction. For addition with a scaled '
                'identity matrix, use e.g. `x + 2 * x.I` instead.'
            )

        if isinstance(y, QArray):
            _check_compatible_dims(self.dims, y.dims)

    def __radd__(self, y: QArrayLike) -> QArray:
        return self.__add__(y)

    def __sub__(self, y: QArrayLike) -> QArray:
        return self + (-y)

    def __rsub__(self, y: QArrayLike) -> QArray:
        return -self + y

    @abstractmethod
    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        pass

    @abstractmethod
    def __rmatmul__(self, y: QArrayLike) -> QArray:
        pass

    @abstractmethod
    def __and__(self, y: QArray) -> QArray:
        pass

    def __pow__(self, power: int | _Metaω) -> QArray:
        # to deal with the x**ω notation from equinox (used in diffrax internals)
        if isinstance(power, _Metaω):
            return _Metaω.__rpow__(power, self)
        else:
            logging.warning(
                'Using the `**` operator performs element-wise power. For matrix '
                'power, use `x @ x @ ... @ x` or `x.powm(power)` instead.'
            )
            return self._pow(power)

    @abstractmethod
    def _pow(self, power: int) -> QArray:
        """Element-wise power of the quantum state."""

    @abstractmethod
    def __getitem__(self, key: int | slice) -> QArray:
        pass


def _check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'QArrays have incompatible dimensions. Got {dims1} and {dims2}.'
        )


def _in_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or any(a % ndim in [ndim - 1, ndim - 2] for a in axis)


def _include_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or (
        ndim - 1 in [a % ndim for a in axis] and ndim - 2 in [a % ndim for a in axis]
    )


# In this file we define an extended array type named `QArrayLike`. Most
# functions in the library take a `QArrayLike` as argument and return a `QArray`.
# `QArrayLike` can be:
# - any numeric type (bool, int, float, complex),
# - a JAX array,
# - a NumPy array,
# - a QuTiP Qobj,
# - a dynamiqs QArray,
# - a nested sequence of these types.
# An object of type `QArrayLike` can be converted to a `QArray` with `asqarray`.

# extended array like type
_QArrayLike = Union[ArrayLike, QArray, Qobj]
# a type alias for nested sequence of _QArrayLike
_NestedQArrayLikeSequence = Sequence[Union[_QArrayLike, '_NestedQArrayLikeSequence']]
# a type alias for any type compatible with asqarray
QArrayLike = Union[_QArrayLike, _NestedQArrayLikeSequence]
