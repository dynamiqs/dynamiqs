from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Any, Union, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar

if TYPE_CHECKING:
    from .dense_qarray import DenseQArray
    from .sparsedia_qarray import SparseDIAQArray
from .layout import Layout

__all__ = ['QArray']


def isqarraylike(x: Any) -> bool:
    r"""Returns True if the input is a qarray-like object.

    Args:
        x: Any object.

    Returns:
        True if `x` is a numeric types (`bool`, `int`, `float`, `complex`), a JAX or
            NumPy array, a QuTiP Qobj, a dynamiqs qarray or any nested sequence of these
            types.

    See also:
        - [`dq.asqarray()`][dynamiqs.asqarray]: converts a qarray-like object into a
            qarray.

    Examples:
        >>> dq.isqarraylike(1)
        True
        >>> dq.isqarraylike(qt.fock(5, 0))
        True
        >>> dq.isqarraylike([qt.fock(5, 0), qt.fock(5, 1)])
        True
    """
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


def _get_dims(x: QArrayLike) -> tuple[int, ...] | None:
    if isinstance(x, Sequence):
        sub_dims = [_get_dims(sub_x) for sub_x in x]
        return sub_dims[0] if all(sd == sub_dims[0] for sd in sub_dims) else None
    if isinstance(x, QArray):
        return x.dims
    elif isinstance(x, Qobj):
        dims = np.max(x.dims, axis=0)
        return tuple(dims.tolist())
    else:
        return None


def _to_numpy(x: QArrayLike) -> np.ndarray:
    if isinstance(x, QArray):
        return x.to_numpy()
    elif isinstance(x, Qobj):
        return np.asarray(x.full())
    elif isinstance(x, Sequence):
        return np.asarray([_to_numpy(sub_x) for sub_x in x])
    else:
        return np.asarray(x)


class QArray(eqx.Module):
    r"""Dynamiqs custom array to represent quantum objects.

    A qarray is a wrapper around the data structure representing a quantum object (a
    ket, a density matrix, an operator, a superoperator, etc.) that offers convenience
    methods.

    There are two types of qarrays:

    - `DenseQArray`: wrapper around JAX arrays, dense representation of the array.
    - `SparseDIAQArray`: Dynamiqs sparse diagonal format, storing only the non-zero
        diagonals.

    Note: Constructing a new qarray from an other array type
        Use the function [`dq.asqarray()`][dynamiqs.asqarray] to create a qarray from a
        qarray-like object. Objects that can be converted to a `QArray` are of type
        `dq.QArrayLike`. This includes all numeric types (`bool`, `int`, `float`,
        `complex`), a JAX or NumPy array, a QuTiP Qobj, a dynamiqs qarray or any nested
        sequence of these types. See also [`dq.isqarraylike()`][dynamiqs.isqarraylike].

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

    Note: Shortcuts methods to use quantum utilities
        Many functions of the library can be called directly on a qarray rather than
        through the functional API. For example, you can use `x.dag()` instead of
        `dq.dag(x)`.

    Here is the list of qarray methods:

    | Method                                                   | Description                                                    |
    |----------------------------------------------------------|----------------------------------------------------------------|
    | [`x.conj()`][dynamiqs.QArray.conj]                       | Returns the element-wise complex conjugate of the qarray.      |
    | `x.dag()`                                                | Alias of [`dq.dag(x)`][dynamiqs.dag].                          |
    | `x.powm()`                                               | Alias of [`dq.powm(x)`][dynamiqs.powm].                        |
    | `x.expm()`                                               | Alias of [`dq.expm(x)`][dynamiqs.expm].                        |
    | `x.cosm()`                                               | Alias of [`dq.cosm(x)`][dynamiqs.cosm].                        |
    | `x.sinm()`                                               | Alias of [`dq.sinm(x)`][dynamiqs.sinm].                        |
    | `x.trace()`                                              | Alias of [`dq.trace(x)`][dynamiqs.trace].                      |
    | `x.ptrace(keep)`                                         | Alias of [`dq.ptrace(x, keep, dims=x.dims)`][dynamiqs.ptrace]. |
    | `x.norm()`                                               | Alias of [`dq.norm(x)`][dynamiqs.norm].                        |
    | `x.unit()`                                               | Alias of [`dq.unit(x)`][dynamiqs.unit].                        |
    | `x.isket()`                                              | Alias of [`dq.isket(x)`][dynamiqs.isket].                      |
    | `x.isbra()`                                              | Alias of [`dq.isbra(x)`][dynamiqs.isbra].                      |
    | `x.isdm()`                                               | Alias of [`dq.isdm(x)`][dynamiqs.isdm].                        |
    | `x.isop()`                                               | Alias of [`dq.isop(x)`][dynamiqs.isop].                        |
    | `x.isherm()`                                             | Alias of [`dq.isherm(x)`][dynamiqs.isherm].                    |
    | `x.toket()`                                              | Alias of [`dq.toket(x)`][dynamiqs.toket].                      |
    | `x.tobra()`                                              | Alias of [`dq.tobra(x)`][dynamiqs.tobra].                      |
    | `x.todm()`                                               | Alias of [`dq.todm(x)`][dynamiqs.todm].                        |
    | `x.proj()`                                               | Alias of [`dq.proj(x)`][dynamiqs.proj].                        |
    | [`x.reshape(*shape)`][dynamiqs.QArray.reshape]           | Returns a reshaped copy of a qarray.                           |
    | [`x.broadcast_to(*shape)`][dynamiqs.QArray.broadcast_to] | Broadcasts a qarray to a new shape.                            |
    | [`x.addscalar(y)`][dynamiqs.QArray.addscalar]            | Adds a scalar.                                                 |
    | [`x.elmul(y)`][dynamiqs.QArray.elmul]                    | Computes the element-wise multiplication.                      |
    | [`x.elpow(power)`][dynamiqs.QArray.elpow]                | Computes the element-wise power.                               |

    There are also several conversion methods available:

    | Method                                                   | Description                                                    |
    |----------------------------------------------------------|----------------------------------------------------------------|
    | `x.to_qutip()`                                           | Alias of [`dq.to_qutip(x, dims=x.dims)`][dynamiqs.to_qutip].   |
    | `x.to_jax()`                                             | Alias of [`dq.to_jax(x)`][dynamiqs.to_jax].                    |
    | `x.to_numpy()`                                           | Alias of [`dq.to_numpy(x)`][dynamiqs.to_numpy].                |
    | [`x.asdense()`][dynamiqs.QArray.asdense]                 | Converts to a dense layout.                                    |
    | [`x.assparsedia()`][dynamiqs.QArray.assparsedia]         | Converts to a sparse diagonal layout.                          |
    """  # noqa: E501

    # Subclasses should implement:
    # - the properties: dtype, layout, shape, mT, _underlying_array
    # - the methods:
    #   - QArray methods: conj, dag, reshape, broadcast_to, ptrace, powm, expm,
    #                     block_until_ready
    #   - returning a JAX array or other: norm, trace, sum, squeeze, _eig, _eigh,
    #                                     _eigvals, _eigvalsh, devices, isherm
    #   - conversion/utils methods: to_qutip, to_jax, __array__, block_until_ready
    #   - special methods: __mul__, __truediv__, __add__, __matmul__, __rmatmul__,
    #                         __and__, addscalar, elmul, elpow, __getitem__

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

    @property
    def _underlying_array(self) -> Array:
        pass

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
    def _eig(self) -> tuple[Array, QArray]:
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

    @abstractmethod
    def asdense(self) -> DenseQArray:
        """Converts to a dense layout.

        Returns:
            A `DenseQArray` object.
        """

    @abstractmethod
    def assparsedia(self) -> SparseDIAQArray:
        """Converts to a sparse diagonal layout.

        Returns:
            A `SparseDIAQArray` object.
        """

    @abstractmethod
    def block_until_ready(self) -> QArray:
        pass

    def __repr__(self) -> str:
        return (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'layout={self.layout}'
        )

    def __neg__(self) -> QArray:
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: ArrayLike) -> QArray:
        if not _is_batched_scalar(y):
            raise NotImplementedError(
                'Element-wise multiplication of two qarrays with the `*` operator is '
                'not supported. For matrix multiplication, use `x @ y`. For '
                'element-wise multiplication, use `x.elmul(y)`.'
            )

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
        if _is_batched_scalar(y):
            raise NotImplementedError(
                'Adding a scalar to a qarray with the `+` operator is not supported. '
                'To add a scaled identity matrix, use `x + scalar * dq.eye(*x.dims)`.'
                ' To add a scalar, use `x.addscalar(scalar)`.'
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
        if _is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')

    @abstractmethod
    def __rmatmul__(self, y: QArrayLike) -> QArray:
        if _is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')

    @abstractmethod
    def __and__(self, y: QArray) -> QArray:
        pass

    def __pow__(self, power: int | _Metaω) -> QArray:
        # to deal with the x**ω notation from equinox (used in diffrax internals)
        if isinstance(power, _Metaω):
            return _Metaω.__rpow__(power, self)

        raise NotImplementedError(
            'Computing the element-wise power of a qarray with the `**` operator is '
            'not supported. For the matrix power, use `x.pomw(power)`. For the '
            'element-wise power, use `x.elpow(power)`.'
        )

    @abstractmethod
    def addscalar(self, y: ArrayLike) -> QArray:
        """Adds a scalar.

        Args:
            y: Scalar to add, whose shape should be broadcastable with the qarray.

        Returns:
            New qarray object resulting from the addition with the scalar.
        """

    @abstractmethod
    def elmul(self, y: ArrayLike) -> QArray:
        """Computes the element-wise multiplication.

        Args:
            y: Array-like object to multiply with element-wise.

        Returns:
            New qarray object resulting from the element-wise multiplication.
        """

    @abstractmethod
    def elpow(self, power: int) -> QArray:
        """Computes the element-wise power.

        Args:
            power: Power to raise to.

        Returns:
            New qarray object with elements raised to the specified power.
        """

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
