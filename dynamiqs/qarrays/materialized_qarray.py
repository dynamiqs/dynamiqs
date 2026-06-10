from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import is_batched_scalar
from .dataarray import DataArray, IndexType
from .layout import Layout
from .qarray import QArray, QArrayLike, check_compatible_dims, isqarraylike, to_jax

__all__ = []


class MaterializedQArray(QArray):
    vectorized: bool = eqx.field(static=True)
    data: DataArray

    def __check_init__(self):

        super().__check_init__()

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
        if not hasattr(self.data, 'ndiags'):
            raise AttributeError(
                f"Attribute 'ndiags' is only defined for sparse diagonal layouts; "
                f'got layout {self.layout!r}.'
            )
        return self.data.ndiags

    # === Array methods delegated to DataArray ===

    def conj(self) -> QArray:
        """Returns the element-wise complex conjugate of the qarray.

        Returns:
            New qarray with element-wise complex conjuguated values.
        """
        return replace(self, data=self.data.conj())

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

    def swapaxes(self, axis1: int, axis2: int) -> QArray:
        """Interchange two axes of a qarray."""
        from .utils import (  # noqa: PLC0415
            _axes_are_batch_axes,
            _normalize_axis,
            _raise_quantum_axis_error,
        )

        axis1 = _normalize_axis(axis1, self.ndim)
        axis2 = _normalize_axis(axis2, self.ndim)
        if axis1 == axis2:
            return self
        if {axis1, axis2} == {self.ndim - 2, self.ndim - 1}:
            return self.mT
        if not _axes_are_batch_axes((axis1, axis2), self.ndim):
            _raise_quantum_axis_error('swapaxes')
        return replace(self, data=self.data.swapaxes(axis1, axis2))

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> QArray:
        """Move axes of a qarray to new positions."""
        from .utils import (  # noqa: PLC0415
            _check_axis_tuples,
            _moveaxis_order,
            _normalize_axes,
            _raise_quantum_axis_error,
        )

        source_axes = _normalize_axes(source, self.ndim)
        destination_axes = _normalize_axes(destination, self.ndim)
        _check_axis_tuples(source_axes, destination_axes)

        axis_order = _moveaxis_order(source_axes, destination_axes, self.ndim)
        identity_order = tuple(range(self.ndim))
        final_swap_order = (*range(self.ndim - 2), self.ndim - 1, self.ndim - 2)
        if axis_order == identity_order:
            return self
        if axis_order == final_swap_order:
            return self.mT
        if not all(axis < self.ndim - 2 for axis in (*source_axes, *destination_axes)):
            _raise_quantum_axis_error('moveaxis')
        return replace(self, data=self.data.moveaxis(source_axes, destination_axes))

    def expand_dims(self, axis: int | Sequence[int]) -> QArray:
        """Expand the shape of a qarray by inserting new axes."""
        from .utils import (  # noqa: PLC0415
            _normalize_insert_axes,
            _raise_quantum_axis_error,
        )

        axes = _normalize_insert_axes(axis, self.ndim)
        out_ndim = self.ndim + len(axes)
        if not all(axis < out_ndim - 2 for axis in axes):
            _raise_quantum_axis_error('expand_dims')
        return replace(self, data=self.data.expand_dims(axes))

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

    def __repr__(self) -> str:
        res = (
            f'QArray: shape={self.shape}, dims={self.dims}, dtype={self.dtype}, '
            f'layout={self.layout}'
        )
        if self.vectorized:
            res += f', vectorized={self.vectorized}'
        res += self.data._repr_extra()
        return res

    def __mul__(self, y: ArrayLike) -> QArray:
        if not is_batched_scalar(y):
            raise NotImplementedError(
                'Element-wise multiplication of two qarrays with the `*` operator is '
                'not supported. For matrix multiplication, use `x @ y`. For '
                'element-wise multiplication, use `x.elmul(y)`.'
            )
        result = self.data * y
        return replace(self, data=result)

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
