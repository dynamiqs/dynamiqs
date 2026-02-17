from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import prod
from typing import Any, TypeAlias, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import is_batched_scalar
from .dataarray import DataArray, IndexType
from .layout import Layout

__all__ = ['QArray']


def isqarraylike(x: Any) -> bool:
    r"""Returns True if the input is a qarray-like.

    Args:
        x: Any object.

    Returns:
        True if `x` is a numeric types (`bool`, `int`, `float`, `complex`), a NumPy or
            JAX array, a Dynamiqs qarray, a QuTiP qobj, or any nested sequence of these
            types.

    See also:
        - [`dq.asqarray()`][dynamiqs.asqarray]: converts a qarray-like into a qarray.

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


def to_jax(x: QArrayLike) -> Array:
    """Convert a qarray-like into a JAX array.

    Args:
        x: Object to convert.

    Returns:
        JAX array.

    Examples:
        >>> dq.to_jax(dq.fock(3, 1))
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_jax([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        Array([[[ 0.+0.j,  1.+0.j],
                [ 1.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[ 0.+0.j,  0.-1.j],
                [ 0.+1.j,  0.+0.j]],
        <BLANKLINE>
               [[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]]], dtype=complex64)
    """
    if isinstance(x, QArray):
        return x.to_jax()
    elif isinstance(x, Qobj):
        return jnp.asarray(x.full())
    elif isinstance(x, Sequence):
        return jnp.asarray([to_jax(sub_x) for sub_x in x])
    else:
        return jnp.asarray(x)


def get_dims(x: QArrayLike) -> tuple[int, ...] | None:
    if isinstance(x, Sequence):
        sub_dims = [get_dims(sub_x) for sub_x in x]
        return sub_dims[0] if all(sd == sub_dims[0] for sd in sub_dims) else None
    if isinstance(x, QArray):
        return x.dims
    elif isinstance(x, Qobj):
        # handle [[3, 2], [1, 1]] or [[1, 1], [3, 2]] when `auto_tidyup_dims=False`
        # or [[3, 2], [1]] or [[1], [3, 2]] when `auto_tidyup_dims=True`
        return tuple(next(dims for dims in x.dims if set(dims) != {1}))
    else:
        return None


def to_numpy(x: QArrayLike) -> np.ndarray:
    """Convert a qarray-like into a NumPy array.

    Args:
        x: Object to convert.

    Returns:
        NumPy array.

    Examples:
        >>> dq.to_numpy(dq.fock(3, 1))
        array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_numpy([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        array([[[ 0.+0.j,  1.+0.j],
                [ 1.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[ 0.+0.j,  0.-1.j],
                [ 0.+1.j,  0.+0.j]],
        <BLANKLINE>
               [[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]]])
    """
    if isinstance(x, QArray):
        return x.to_numpy()
    elif isinstance(x, Qobj):
        return np.asarray(x.full())
    elif isinstance(x, Sequence):
        return np.asarray([to_numpy(sub_x) for sub_x in x])
    else:
        return np.asarray(x)


class QArray(eqx.Module):
    r"""Dynamiqs custom array to represent quantum objects.

    A qarray is a wrapper around the data structure representing a quantum object (a
    ket, a density matrix, an operator, a superoperator, etc.) that offers convenience
    methods.

    The underlying data can be in two formats:

    - Dense: wrapper around JAX arrays, dense representation of the array.
    - SparseDIA: Dynamiqs sparse diagonal format, storing only the non-zero diagonals.

    Note: Constructing a new qarray from an other array type
        Use the function [`dq.asqarray()`][dynamiqs.asqarray] to create a qarray from a
        qarray-like. Objects that can be converted to a `QArray` are of type
        `dq.QArrayLike`. This includes all numeric types (`bool`, `int`, `float`,
        `complex`), a NumPy or JAX array, a Dynamiqs qarray, a QuTiP qobj, or any nested
        sequence of these types. See also [`dq.isqarraylike()`][dynamiqs.isqarraylike].

    Attributes:
        dtype (numpy.dtype): Data type.
        shape (tuple of ints): Shape.
        ndim (int): Number of dimensions in the shape.
        layout (Layout): Data layout, either `dq.dense` or `dq.dia`.
        dims (tuple of ints): Hilbert space dimension of each subsystem.
        mT (qarray): Returns the qarray transposed over its last two dimensions.
        vectorized (bool): Whether the underlying object is non-vectorized (ket, bra
            or operator) or vectorized (operator in vector form or superoperator in
            matrix form).

    Note: Arithmetic operation support
        Qarrays support basic arithmetic operations `-, +, *, /, @` with other
        qarray-likes.

    Note: Shortcuts methods to use quantum utilities
        Many functions of the library can be called directly on a qarray rather than
        through the functional API. For example, you can use `x.dag()` instead of
        `dq.dag(x)`.

    Here is the list of qarray methods:

    | Method                                                   | Description                                                    |
    |----------------------------------------------------------|----------------------------------------------------------------|
    | [`x.conj()`][dynamiqs.qarrays.qarray.QArray.conj]        | Returns the element-wise complex conjugate of the qarray.      |
    | `x.dag()`                                                | Alias of [`dq.dag(x)`][dynamiqs.dag].                          |
    | `x.powm()`                                               | Alias of [`dq.powm(x)`][dynamiqs.powm].                        |
    | `x.expm()`                                               | Alias of [`dq.expm(x)`][dynamiqs.expm].                        |
    | `x.cosm()`                                               | Alias of [`dq.cosm(x)`][dynamiqs.cosm].                        |
    | `x.sinm()`                                               | Alias of [`dq.sinm(x)`][dynamiqs.sinm].                        |
    | `x.signm()`                                              | Alias of [`dq.signm(x)`][dynamiqs.signm].                      |
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
    | [`x.reshape(*shape)`][dynamiqs.qarrays.qarray.QArray.reshape] | Returns a reshaped copy of a qarray.                           |
    | [`x.broadcast_to(*shape)`][dynamiqs.qarrays.qarray.QArray.broadcast_to] | Broadcasts a qarray to a new shape.                            |
    | [`x.addscalar(y)`][dynamiqs.qarrays.qarray.QArray.addscalar] | Adds a scalar.                                                 |
    | [`x.elmul(y)`][dynamiqs.qarrays.qarray.QArray.elmul]     | Computes the element-wise multiplication.                      |
    | [`x.elpow(power)`][dynamiqs.qarrays.qarray.QArray.elpow] | Computes the element-wise power.                               |

    There are also several conversion methods available:

    | Method                                                   | Description                                                    |
    |----------------------------------------------------------|----------------------------------------------------------------|
    | `x.to_qutip()`                                           | Alias of [`dq.to_qutip(x, dims=x.dims)`][dynamiqs.to_qutip].   |
    | `x.to_jax()`                                             | Alias of [`dq.to_jax(x)`][dynamiqs.to_jax].                    |
    | `x.to_numpy()`                                           | Alias of [`dq.to_numpy(x)`][dynamiqs.to_numpy].                |
    | [`x.asdense()`][dynamiqs.qarrays.qarray.QArray.asdense]  | Converts to a dense layout.                                    |
    | [`x.assparsedia()`][dynamiqs.qarrays.qarray.QArray.assparsedia] | Converts to a sparse diagonal layout.                          |
    """  # noqa: E501

    dims: tuple[int, ...] = eqx.field(static=True)
    vectorized: bool = eqx.field(static=True)
    data: DataArray

    # Increase __array_priority__ to ensure that a qarray is always returned during an
    # arithmetic operation with a NumPy array. In JAX, it is set to 100 for arrays, and
    # in NumPy it is set to 0.
    __array_priority__ = 200

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
        shape = self.data.shape
        allowed_shapes = (prod(self.dims), prod(self.dims) ** 2)
        if not (shape[-1] in allowed_shapes or shape[-2] in allowed_shapes):
            raise ValueError(
                'Argument `dims` must be compatible with the shape of the qarray, but '
                f'got dims {self.dims} and shape {shape}.'
            )

    # === Properties delegated to DataArray ===

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def layout(self) -> Layout:
        return self.data.layout

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def mT(self) -> QArray:
        return replace(self, data=self.data.mT)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def ndiags(self) -> int:
        """Number of stored diagonals (only for sparse diagonal layout)."""
        return self.data.ndiags

    # === Array methods delegated to DataArray ===

    def conj(self) -> QArray:
        """Returns the element-wise complex conjugate of the qarray.

        Returns:
            New qarray with element-wise complex conjuguated values.
        """
        return replace(self, data=self.data.conj())

    def dag(self) -> QArray:
        return self.mT.conj()

    def reshape(self, *shape: int) -> QArray:
        """Returns a reshaped copy of a qarray.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New qarray with the given shape.
        """
        return replace(self, data=self.data.reshape(*shape))

    def _reshape_unchecked(self, *shape: int) -> QArray:
        return replace(self, data=self.data._reshape_unchecked(*shape))

    def broadcast_to(self, *shape: int) -> QArray:
        """Broadcasts a qarray to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New qarray with the given shape.
        """
        return replace(self, data=self.data.broadcast_to(*shape))

    def powm(self, n: int) -> QArray:
        return replace(self, data=self.data.powm(n))

    def expm(self, *, max_squarings: int = 16) -> QArray:
        return replace(self, data=self.data.expm(max_squarings=max_squarings))

    def norm(self, *, psd: bool = False) -> Array:
        return self.data.norm(psd=psd)

    def trace(self) -> Array:
        return self.data.trace()

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        result = self.data.sum(axis=axis)
        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def mean(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        numerator = self.sum(axis=axis)
        if axis is None:
            denominator = prod(self.shape)
        elif isinstance(axis, int):
            denominator = self.shape[axis % self.ndim]
        else:
            denominator = prod(self.shape[i % self.ndim] for i in axis)
        return numerator / denominator

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        result = self.data.squeeze(axis=axis)
        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def _eig(self) -> tuple[Array, QArray]:
        evals, evecs = self.data._eig()
        return evals, replace(self, data=evecs)

    def _eigh(self) -> tuple[Array, Array]:
        return self.data._eigh()

    def _eigvals(self) -> Array:
        return self.data._eigvals()

    def _eigvalsh(self) -> Array:
        return self.data._eigvalsh()

    def devices(self) -> set[Device]:
        return self.data.devices()

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return self.data.isherm(rtol=rtol, atol=atol)

    def block_until_ready(self) -> QArray:
        self.data.block_until_ready()
        return self

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        from ..utils.general import ptrace  # noqa: PLC0415

        return ptrace(self.data.to_jax(), keep, self.dims)

    def cosm(self) -> QArray:
        from ..utils import cosm  # noqa: PLC0415

        return cosm(self)

    def sinm(self) -> QArray:
        from ..utils import sinm  # noqa: PLC0415

        return sinm(self)

    def signm(self) -> QArray:
        from ..utils import signm  # noqa: PLC0415

        return signm(self)

    def unit(self, *, psd: bool = False) -> QArray:
        return self / self.norm(psd=psd)[..., None, None]

    def isket(self) -> bool:
        from ..utils import isket  # noqa: PLC0415

        return isket(self)

    def isbra(self) -> bool:
        from ..utils import isbra  # noqa: PLC0415

        return isbra(self)

    def isdm(self) -> bool:
        from ..utils import isdm  # noqa: PLC0415

        return isdm(self)

    def isop(self) -> bool:
        from ..utils import isop  # noqa: PLC0415

        return isop(self)

    def toket(self) -> QArray:
        from ..utils import toket  # noqa: PLC0415

        return toket(self)

    def tobra(self) -> QArray:
        from ..utils import tobra  # noqa: PLC0415

        return tobra(self)

    def todm(self) -> QArray:
        from ..utils import todm  # noqa: PLC0415

        return todm(self)

    def proj(self) -> QArray:
        from ..utils import proj  # noqa: PLC0415

        return proj(self)

    # === Conversion methods ===

    def to_qutip(self) -> Qobj | list[Qobj]:
        from .dense_dataarray import array_to_qobj_list  # noqa: PLC0415

        return array_to_qobj_list(self.data.to_jax(), self.dims)

    def to_jax(self) -> Array:
        return self.data.to_jax()

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.data.__array__(dtype=dtype, copy=copy)

    def asdense(self) -> QArray:
        """Converts to a dense layout.

        Returns:
            A qarray with dense data layout.
        """
        return replace(self, data=self.data.asdense())

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
        """Converts to a sparse diagonal layout.

        Args:
            offsets: Offsets of the stored diagonals. If `None`, offsets are determined
                automatically from the matrix structure. This argument can also be
                explicitly specified to ensure compatibility with JAX transformations,
                which require static offset values.

        Returns:
            A qarray with sparse diagonal data layout.
        """
        return replace(self, data=self.data.assparsedia(offsets))

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError('len() of unsized object') from err

    # === Repr ===

    def __repr__(self) -> str:
        res = (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'layout={self.layout}'
        )
        if self.vectorized:
            res += f', vectorized={self.vectorized}'
        res += self.data._repr_extra()
        return res

    # === Arithmetic operations ===

    def __neg__(self) -> QArray:
        return self * (-1)

    def __mul__(self, y: ArrayLike) -> QArray:
        if not is_batched_scalar(y):
            raise NotImplementedError(
                'Element-wise multiplication of two qarrays with the `*` operator is '
                'not supported. For matrix multiplication, use `x @ y`. For '
                'element-wise multiplication, use `x.elmul(y)`.'
            )
        result = self.data * y
        return replace(self, data=result)

    def __rmul__(self, y: QArrayLike) -> QArray:
        return self * y

    def __truediv__(self, y: ArrayLike) -> QArray:
        return self * (1 / y)

    def __rtruediv__(self, y: QArrayLike) -> QArray:
        raise NotImplementedError(
            'Division by a qarray with the `/` operator is not supported.'
        )

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __add__(self, y: QArrayLike) -> QArray:
        if isinstance(y, int | float) and y == 0:
            return self

        if is_batched_scalar(y):
            raise NotImplementedError(
                'Adding a scalar to a qarray with the `+` operator is not supported. '
                'To add a scaled identity matrix, use `x + scalar * dq.eye_like(x)`.'
                ' To add a scalar, use `x.addscalar(scalar)`.'
            )

        if isinstance(y, QArray):
            check_compatible_dims(self.dims, y.dims)
            result = self.data + y.data
        elif isqarraylike(y):
            result = self.data + to_jax(y)
        else:
            return NotImplemented

        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def __radd__(self, y: QArrayLike) -> QArray:
        return self.__add__(y)

    def __sub__(self, y: QArrayLike) -> QArray:
        return self + (-y)

    def __rsub__(self, y: QArrayLike) -> QArray:
        return -self + y

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        if isinstance(y, QArray):
            check_compatible_dims(self.dims, y.dims)
            y_data = y.data
        elif is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')
        elif isqarraylike(y):
            y_data = to_jax(y)
        else:
            return NotImplemented

        result = self.data @ y_data
        if result is NotImplemented:
            # try reverse dispatch
            if hasattr(y_data, '__rmatmul__'):
                result = y_data.__rmatmul__(self.data)
            # if still NotImplemented, raise it
            if result is NotImplemented:
                return NotImplemented

        # bra @ ket → scalar
        if (
            isinstance(y, QArray)
            and self.isbra()
            and y.isket()
            and isinstance(result, DataArray)
        ):
            result = result.to_jax()

        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        if isinstance(y, QArray):
            check_compatible_dims(self.dims, y.dims)
            y_data = y.data
        elif is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')
        elif isqarraylike(y):
            y_data = to_jax(y)
        else:
            return NotImplemented

        # y_data @ self.data
        if isinstance(y_data, DataArray):
            result = y_data @ self.data
        else:
            # y_data is a raw array; use DataArray's __rmatmul__
            result = self.data.__rmatmul__(y_data)

        if result is NotImplemented:
            return NotImplemented

        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def __and__(self, y: QArray) -> QArray:
        if not isinstance(y, QArray):
            return NotImplemented

        result = self.data & y.data
        if result is NotImplemented:
            # try reverse dispatch
            if hasattr(y.data, '__rand__'):
                result = y.data.__rand__(self.data)
            # if still NotImplemented, raise it
            if result is NotImplemented:
                return NotImplemented

        new_dims = self.dims + y.dims
        return replace(self, dims=new_dims, data=result)

    def __pow__(self, power: int | _Metaω) -> QArray:
        # to deal with the x**ω notation from equinox (used in diffrax internals)
        if isinstance(power, _Metaω):
            return _Metaω.__rpow__(power, self)

        raise NotImplementedError(
            'Computing the element-wise power of a qarray with the `**` operator is '
            'not supported. For the matrix power, use `x.powm(power)`. For the '
            'element-wise power, use `x.elpow(power)`.'
        )

    def addscalar(self, y: ArrayLike) -> QArray:
        """Adds a scalar.

        Args:
            y: Scalar to add, whose shape should be broadcastable with the qarray.

        Returns:
            New qarray resulting from the addition with the scalar.
        """
        return replace(self, data=self.data + jnp.asarray(y))

    def elmul(self, y: QArrayLike) -> QArray:
        """Computes the element-wise multiplication.

        Args:
            y: Qarray-like to multiply with element-wise.

        Returns:
            New qarray resulting from the element-wise multiplication.
        """
        if isinstance(y, QArray):
            check_compatible_dims(self.dims, y.dims)
            result = self.data * y.data
        elif isqarraylike(y):
            result = self.data * to_jax(y)
        else:
            return NotImplemented

        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, DataArray):
            return replace(self, data=result)
        return result

    def elpow(self, power: int) -> QArray:
        """Computes the element-wise power.

        Args:
            power: Power to raise to.

        Returns:
            New qarray with elements raised to the specified power.
        """
        return replace(self, data=self.data**power)

    def __getitem__(self, key: IndexType) -> QArray:
        result = self.data[key]
        return replace(self, data=result)


def check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'Qarrays have incompatible Hilbert space dimensions. '
            f'Got {dims1} and {dims2}.'
        )


# In this file we define an extended array type named `QArrayLike`. Most
# functions in the library take a `QArrayLike` as argument and return a `QArray`.
# `QArrayLike` can be:
# - any numeric type (bool, int, float, complex),
# - a NumPy array,
# - a JAX array,
# - a dynamiqs `QArray`,
# - a QuTiP Qobj,
# - a nested sequence of these types.
# An object of type `QArrayLike` can be converted to a `QArray` with `asqarray`.

# extended array-like type
_QArrayLike = ArrayLike | QArray | Qobj
# a type alias for nested sequence of `_QArrayLike`
_NestedQArrayLikeSequence = Sequence[_QArrayLike | '_NestedQArrayLikeSequence']
# a type alias for any type compatible with asqarray
QArrayLike: TypeAlias = _QArrayLike | _NestedQArrayLikeSequence
