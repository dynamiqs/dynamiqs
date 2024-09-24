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
    elif isinstance(x, list):
        return all(isqarraylike(sub_x) for sub_x in x)
    return False


def _asjaxarray(x: QArrayLike) -> Array:
    if isinstance(x, QArray):
        return x.asjaxarray()
    elif isinstance(x, Sequence):
        return jnp.asarray([_asjaxarray(sub_x) for sub_x in x])
    else:
        return jnp.asarray(x)


class QArray(eqx.Module):
    r"""Quantum array object. DenseQArray is a wrapper around JAX arrays. It offers
    convenience methods to handle more easily quantum states
    (bras, kets and density matrices) and quantum operators.
    If you come from QuTiP, this is the equivalent of the Qobj class.
    """

    # Subclasses should implement:
    # - the properties: dtype, layout, shape, mT
    # - the methods:
    #   - QArray methods: conj, dag, reshape, broadcast_to, ptrace, powm, expm,
    #                     _abs
    #   - returning a JAX array or other: norm, trace, sum, squeeze, _eigh, _eigvals,
    #                                     _eigvalsh, devices, isherm
    #   - conversion methods: asqobj, asjaxarray, __array__
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
        """Return the data type of the quantum state.

        Returns:
             The data type of the quantum state.
        """

    @property
    @abstractmethod
    def layout(self) -> Layout:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the quantum state.

        Returns:
            The shape of the quantum state.
        """

    @property
    @abstractmethod
    def mT(self) -> QArray:
        pass

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the quantum state.

        Returns:
            The number of dimensions of the quantum state.
        """
        return len(self.shape)

    @abstractmethod
    def conj(self) -> QArray:
        """Returns the conjugate of the quantum state.

        Returns:
            The conjugate of the quantum state.
        """

    def dag(self) -> QArray:
        """Returns the dagger of the quantum state.

        Returns:
            The dagger of the quantum state.
        """
        return self.mT.conj()

    @abstractmethod
    def reshape(self, *shape: int) -> QArray:
        """Returns the reshaped quantum state.

        Args:
            shape: New shape of the quantum state.

        Returns:
            The reshaped quantum state.
        """

    @abstractmethod
    def broadcast_to(self, *shape: int) -> QArray:
        """Returns the broadcast the quantum state.

        Returns:
            The broadcast quantum state.
        """

    @abstractmethod
    def ptrace(self, *keep: int) -> QArray:
        """Returns the partial trace of the quantum state.

        Args:
            keep: Dimensions to keep.

        Returns:
            The partial trace of the quantum state.
        """

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
        """Returns the normalized the quantum state.

        Returns:
            The normalized quantum state.
        """
        return self / self.norm()[..., None, None]

    @abstractmethod
    def norm(self) -> Array:
        """Returns the norm of the quantum state.

        Returns:
            The norm of the quantum state.
        """

    @abstractmethod
    def trace(self) -> Array:
        pass

    def entropy_vn(self) -> Array:
        from ..utils import entropy_vn

        return entropy_vn(self)

    @abstractmethod
    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    @abstractmethod
    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
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
        pass

    def isket(self) -> bool:
        """Returns the check if the quantum state is a ket.

        Returns:
            True if the quantum state is a ket, False otherwise.
        """
        from ..utils import isket

        return isket(self)

    def isbra(self) -> bool:
        """Returns the check if the quantum state is a bra.

        Returns:
            True if the quantum state is a bra, False otherwise.
        """
        from ..utils import isbra

        return isbra(self)

    def isdm(self) -> bool:
        """Returns the check if the quantum state is a density matrix.

        Returns:
            True if the quantum state is a density matrix, False otherwise.
        """
        from ..utils import isdm

        return isdm(self)

    def isop(self) -> bool:
        """Returns the check if the quantum state is an operator.

        Returns:
            True if the quantum state is an operator, False otherwise.
        """
        from ..utils import isop

        return isop(self)

    @abstractmethod
    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Returns the check if the quantum state is Hermitian.

        Returns:
            True if the quantum state is Hermitian, False otherwise.
        """

    def toket(self) -> QArray:
        """Convert the quantum state to a ket.

        Returns:
            The ket representation of the quantum state.
        """
        from ..utils import toket

        return toket(self)

    def tobra(self) -> QArray:
        """Convert the quantum state to a bra.

        Returns:
            The bra representation of the quantum state.
        """
        from ..utils import tobra

        return tobra(self)

    def todm(self) -> QArray:
        """Convert the quantum state to a density matrix.

        Returns:
            The density matrix representation of the quantum state.
        """
        from ..utils import todm

        return todm(self)

    def proj(self) -> QArray:
        """Projector of the quantum state.

        Returns:
            The projector of the quantum state.
        """
        from ..utils import proj

        return proj(self)

    @abstractmethod
    def asqobj(self) -> Qobj | list[Qobj]:
        """Convert the quantum state to a QuTiP object.

        Returns:
            A QuTiP object representation of the quantum state.
        """

    @abstractmethod
    def asjaxarray(self) -> Array:
        """Convert the quantum state to a JAX array.

        Returns:
            The JAX array representation of the quantum state.
        """

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError('len() of unsized object') from err

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        pass

    def asnparray(self) -> np.ndarray:
        """Convert the quantum state to a NumPy array.

        Returns:
            The NumPy array representation of the quantum state.
        """
        return np.asarray(self)

    def __repr__(self) -> str:
        return (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'layout={self.layout}'
        )

    def __neg__(self) -> QArray:
        """Negate the quantum state."""
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: QArrayLike) -> QArray:
        """Element-wise multiplication with a scalar or an array."""
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
        """Element-wise multiplication with a scalar or an array on the right."""
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
        """Element-wise addition with a scalar or an array."""
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
        """Element-wise addition with a scalar or an array on the right."""
        return self.__add__(y)

    def __sub__(self, y: QArrayLike) -> QArray:
        """Element-wise subtraction with a scalar or an array."""
        return self + (-y)

    def __rsub__(self, y: QArrayLike) -> QArray:
        """Element-wise subtraction with a scalar or an array on the right."""
        return -self + y

    @abstractmethod
    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        """Matrix multiplication with another quantum state or JAX array."""

    @abstractmethod
    def __rmatmul__(self, y: QArrayLike) -> QArray:
        """Matrix multiplication with another quantum state or JAX array
        on the right.
        """

    @abstractmethod
    def __and__(self, y: QArray) -> QArray:
        """Tensor product between two quantum states."""

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
# - a nested list of these types.
# An object of type `QArrayLike` can be converted to a `QArray` with `asqarray`.

# extended array like type
_QArrayLike = Union[ArrayLike, QArray, Qobj]
# a type alias for nested list of _QArrayLike
_NestedQArrayLikeList = list[Union[_QArrayLike, '_NestedQArrayLikeList']]
# a type alias for any type compatible with asqarray
QArrayLike = Union[_QArrayLike, _NestedQArrayLikeList]
