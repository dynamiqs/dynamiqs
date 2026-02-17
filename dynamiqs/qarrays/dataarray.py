from __future__ import annotations

from abc import abstractmethod
from math import prod
from types import EllipsisType
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jax import Array, Device
from jaxtyping import ArrayLike

from .layout import Layout

if TYPE_CHECKING:
    from .dense_dataarray import DenseDataArray
    from .sparsedia_dataarray import SparseDIADataArray

IndexType: TypeAlias = (
    int | slice | EllipsisType | None | tuple[int | slice | EllipsisType | None, ...]
)


class DataArray(eqx.Module):
    """Internal base class for array data storage (dense or sparse diagonal).

    This class handles JAX-related utilities and array manipulation. It does not know
    about quantum semantics (dims, vectorized, etc.). See QArray for the public API.
    """

    # Subclasses should implement:
    # - properties: dtype, layout, shape, mT
    # - methods:
    #   - array methods: conj, _reshape_unchecked, broadcast_to, powm, expm,
    #                    block_until_ready
    #   - returning a JAX array or other: norm, trace, sum, squeeze, _eig, _eigh,
    #                                     _eigvals, _eigvalsh, devices, isherm
    #   - conversion/utils: to_jax, __array__, block_until_ready
    #   - arithmetic: __mul__, __add__, __matmul__, __rmatmul__, __and__,
    #                 __pow__, __getitem__
    #   - repr: _repr_extra

    # Priority for matmul dispatch between dense and sparse.
    _matmul_priority: ClassVar[int] = 0

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
    def mT(self) -> DataArray:
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @abstractmethod
    def conj(self) -> DataArray:
        """Returns the element-wise complex conjugate."""

    def dag(self) -> DataArray:
        return self.mT.conj()

    def reshape(self, *shape: int) -> DataArray:
        """Returns a reshaped copy.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New data array with the given shape.
        """
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot reshape to shape {shape} because the last two dimensions do '
                f'not match current shape dimensions, {self.shape}.'
            )
        return self._reshape_unchecked(*shape)

    @abstractmethod
    def _reshape_unchecked(self, *shape: int) -> DataArray:
        """Does the heavy-lifting for `reshape` but skips all checks."""

    @abstractmethod
    def broadcast_to(self, *shape: int) -> DataArray:
        """Broadcasts to a new shape."""

    @abstractmethod
    def powm(self, n: int) -> DataArray:
        pass

    @abstractmethod
    def expm(self, *, max_squarings: int = 16) -> DataArray:
        pass

    @abstractmethod
    def norm(self, *, psd: bool = False) -> Array:
        pass

    @abstractmethod
    def trace(self) -> Array:
        pass

    @abstractmethod
    def sum(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        pass

    def mean(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        numerator = self.sum(axis=axis)
        if axis is None:
            denominator = prod(self.shape)
        elif isinstance(axis, int):
            denominator = self.shape[axis % self.ndim]
        else:
            denominator = prod(self.shape[i % self.ndim] for i in axis)
        return numerator / denominator

    @abstractmethod
    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        pass

    @abstractmethod
    def _eig(self) -> tuple[Array, DataArray]:
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

    @abstractmethod
    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        pass

    @abstractmethod
    def to_jax(self) -> Array:
        pass

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self)

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        pass

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError('len() of unsized object') from err

    @abstractmethod
    def asdense(self) -> DenseDataArray:
        """Converts to a dense layout."""

    @abstractmethod
    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> SparseDIADataArray:
        """Converts to a sparse diagonal layout."""

    @abstractmethod
    def block_until_ready(self) -> DataArray:
        pass

    @abstractmethod
    def _repr_extra(self) -> str:
        """Return extra repr info (ndiags, data display) appended to QArray repr."""

    # === Arithmetic operations ===

    @abstractmethod
    def __mul__(self, y: DataArrayLike) -> DataArray:
        pass

    def __neg__(self) -> DataArray:
        return self * (-1)

    def __rmul__(self, y: DataArrayLike) -> DataArray:
        return self * y

    def __truediv__(self, y: ArrayLike) -> DataArray:
        return self * (1 / y)

    @abstractmethod
    def __add__(self, y: DataArrayLike) -> DataArray:
        pass

    def __radd__(self, y: DataArrayLike) -> DataArray:
        return self.__add__(y)

    def __sub__(self, y: DataArrayLike) -> DataArray:
        return self + (-y)

    def __rsub__(self, y: DataArrayLike) -> DataArray:
        return -self + y

    @abstractmethod
    def __matmul__(self, y: DataArrayLike) -> DataArray | Array:
        pass

    @abstractmethod
    def __rmatmul__(self, y: DataArrayLike) -> DataArray:
        pass

    @abstractmethod
    def __and__(self, y: DataArray) -> DataArray:
        pass

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    @abstractmethod
    def __pow__(self, power: int | _Metaω) -> DataArray:
        pass

    @abstractmethod
    def __getitem__(self, key: IndexType) -> DataArray:
        pass


def in_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or any(a % ndim in [ndim - 1, ndim - 2] for a in axis)


def include_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or (
        ndim - 1 in [a % ndim for a in axis] and ndim - 2 in [a % ndim for a in axis]
    )


# A type alias for DataArray or raw JAX/NumPy array-like values.
DataArrayLike: TypeAlias = DataArray | ArrayLike
