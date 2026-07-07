from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
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

_IndexItemType: TypeAlias = int | slice | ArrayLike | EllipsisType | None
IndexType: TypeAlias = _IndexItemType | tuple[_IndexItemType, ...]


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

    def swapaxes(self, axis1: int, axis2: int) -> DataArray:
        """Interchanges two axes."""
        axis1 = _normalize_axis(axis1, self.ndim)
        axis2 = _normalize_axis(axis2, self.ndim)
        if axis1 == axis2:
            return self
        if {axis1, axis2} == {self.ndim - 2, self.ndim - 1}:
            return self.mT
        if in_last_two_dims((axis1, axis2), self.ndim):
            _raise_matrix_axis_error('swapaxes')
        return self._swapaxes_unchecked(axis1, axis2)

    @abstractmethod
    def _swapaxes_unchecked(self, axis1: int, axis2: int) -> DataArray:
        """Does the heavy-lifting for `swapaxes` but skips all checks."""

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> DataArray:
        """Moves axes to new positions."""
        source_axes = (source,) if isinstance(source, int) else tuple(source)
        source_axes = tuple(_normalize_axis(axis, self.ndim) for axis in source_axes)
        destination_axes = (
            (destination,) if isinstance(destination, int) else tuple(destination)
        )
        destination_axes = tuple(
            _normalize_axis(axis, self.ndim) for axis in destination_axes
        )
        _check_axis_tuples(source_axes, destination_axes)

        axis_order = _moveaxis_order(source_axes, destination_axes, self.ndim)
        identity_order = tuple(range(self.ndim))
        final_swap_order = (*range(self.ndim - 2), self.ndim - 1, self.ndim - 2)
        if axis_order == identity_order:
            return self
        if axis_order == final_swap_order:
            return self.mT
        if set(axis_order[:-2]) != set(range(self.ndim - 2)):
            _raise_matrix_axis_error('moveaxis')

        batch_source = tuple(range(self.ndim - 2))
        batch_destination = tuple(axis_order[:-2].index(axis) for axis in batch_source)
        if axis_order[-2:] == identity_order[-2:]:
            return self._moveaxis_unchecked(batch_source, batch_destination)
        elif axis_order[-2:] == final_swap_order[-2:]:
            return self.mT._moveaxis_unchecked(batch_source, batch_destination)
        else:
            _raise_matrix_axis_error('moveaxis')
        raise AssertionError('unreachable')

    @abstractmethod
    def _moveaxis_unchecked(
        self, source: tuple[int, ...], destination: tuple[int, ...]
    ) -> DataArray:
        """Does the heavy-lifting for `moveaxis` but skips all checks."""

    def expand_dims(self, axis: int | Sequence[int]) -> DataArray:
        """Expands the shape by inserting new axes."""
        axes = _normalize_insert_axes(axis, self.ndim)
        if in_last_two_dims(axes, self.ndim + len(axes)):
            _raise_matrix_axis_error('expand_dims')
        return self._expand_dims_unchecked(axes)

    @abstractmethod
    def _expand_dims_unchecked(self, axis: tuple[int, ...]) -> DataArray:
        """Does the heavy-lifting for `expand_dims` but skips all checks."""

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
        return self * (1 / jnp.asarray(y))

    @abstractmethod
    def __add__(self, y: DataArrayLike) -> DataArray:
        pass

    def __radd__(self, y: DataArrayLike) -> DataArray:
        return self.__add__(y)

    def __sub__(self, y: DataArrayLike) -> DataArray:
        if not isinstance(y, DataArray):
            y = jnp.asarray(y)
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
    def __getitem__(self, key: IndexType) -> DataArray | Array:
        pass


def in_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or any(a % ndim in [ndim - 1, ndim - 2] for a in axis)


def include_last_two_dims(axis: int | tuple[int, ...] | None, ndim: int) -> bool:
    axis = (axis,) if isinstance(axis, int) else axis
    return axis is None or (
        ndim - 1 in [a % ndim for a in axis] and ndim - 2 in [a % ndim for a in axis]
    )


def key_touches_last_two_dims(key: IndexType, ndim: int) -> bool:
    # returns True if `key` modifies either of the last two dimensions of an array of
    # `ndim` dimensions: indexing them with anything other than a full slice, or
    # inserting a new axis between or after them
    items: tuple = key if isinstance(key, tuple) else (key,)

    # first, expand any ellipsis into full slices for the missing dimensions
    ellipsis_indices = [i for i, k in enumerate(items) if k is Ellipsis]
    if len(ellipsis_indices) > 0:
        i = ellipsis_indices[0]
        n_consumed = sum(_consumed_ndim(k) for k in items if k is not Ellipsis)
        fill = (slice(None),) * max(ndim - n_consumed, 0)
        items = items[:i] + fill + items[i + 1 :]

    # then, walk the key while tracking the axis each item applies to
    axis = 0
    for k in items:
        n = _consumed_ndim(k)
        if n == 0:
            # newaxis-like item: breaks the matrix structure if the new axis is
            # inserted between or after the last two dimensions
            if axis >= ndim - 1:
                return True
            continue
        # the item consumes axes [axis, axis + n): non-trivial if it overlaps the
        # last two dimensions
        if axis + n > ndim - 2 and not _is_full_slice(k):
            return True
        axis += n

    return False


def _consumed_ndim(k: object) -> int:
    # number of array dimensions consumed by a single index item
    if k is None or isinstance(k, bool):
        return 0  # newaxis-like: inserts a dimension, consumes none
    if isinstance(k, np.ndarray | np.generic | Array) and k.dtype == bool:
        return k.ndim  # boolean mask: consumes as many dimensions as its ndim
    return 1  # int, slice or integer array (advanced indexing consumes one dim)


def _is_full_slice(k: object) -> bool:
    return (
        isinstance(k, slice)
        and k.start in (None, 0)
        and k.stop is None
        and k.step in (None, 1)
    )


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f'axis {axis} is out of bounds for array of dimension {ndim}')
    return axis % ndim


def _normalize_insert_axes(axis: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_ndim = ndim + len(axes)
    normalized_axes = tuple(_normalize_axis(ax, out_ndim) for ax in axes)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError(f'Repeated axis found in axes to insert. Got `axis={axis}`.')
    return normalized_axes


def _check_axis_tuples(source: tuple[int, ...], destination: tuple[int, ...]) -> None:
    if len(source) != len(destination):
        raise ValueError(
            '`source` and `destination` arguments must have the same number of axes.'
        )
    if len(set(source)) != len(source):
        raise ValueError(f'Repeated axis found in `source` argument. Got {source}.')
    if len(set(destination)) != len(destination):
        raise ValueError(
            f'Repeated axis found in `destination` argument. Got {destination}.'
        )


def _moveaxis_order(
    source: tuple[int, ...], destination: tuple[int, ...], ndim: int
) -> tuple[int, ...]:
    order = [axis for axis in range(ndim) if axis not in source]
    for destination_axis, source_axis in sorted(zip(destination, source, strict=True)):
        order.insert(destination_axis, source_axis)
    return tuple(order)


def _raise_matrix_axis_error(operation: str) -> None:
    raise ValueError(
        f'`{operation}` can only manipulate batching dimensions of a data array; the '
        'final two dimensions are matrix dimensions. Use `swapaxes(-1, -2)` or '
        '`.mT` to transpose those final dimensions.'
    )


# A type alias for DataArray or raw JAX/NumPy array-like values.
DataArrayLike: TypeAlias = DataArray | ArrayLike
