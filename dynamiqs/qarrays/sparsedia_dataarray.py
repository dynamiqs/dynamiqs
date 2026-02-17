from __future__ import annotations

import re
import warnings
from dataclasses import replace
from typing import ClassVar, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal._omega import _Metaω  # noqa: PLC2403
from jaxtyping import Array, ArrayLike

from .dataarray import DataArray, IndexType, in_last_two_dims, include_last_two_dims
from .dense_dataarray import DenseDataArray
from .layout import Layout, dia
from .sparsedia_primitives import (
    add_sparsedia_sparsedia,
    and_sparsedia_sparsedia,
    broadcast_sparsedia,
    matmul_array_sparsedia,
    matmul_sparsedia_array,
    matmul_sparsedia_sparsedia,
    mul_sparsedia_array,
    mul_sparsedia_sparsedia,
    powm_sparsedia,
    reshape_sparsedia,
    shape_sparsedia,
    sparsedia_to_array,
    trace_sparsedia,
    transpose_sparsedia,
)

__all__ = ['SparseDIADataArray']


class SparseDIADataArray(DataArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array = eqx.field(converter=jnp.asarray)

    _matmul_priority: ClassVar[int] = 10

    def __check_init__(self):
        # check diags and offsets have the right type and shape before compressing them
        if not isinstance(self.offsets, tuple):
            raise TypeError(
                'Argument `offsets` of `SparseDIADataArray` must be a tuple but got '
                f'{type(self.offsets)}'
            )

        if self.diags.ndim < 2 or self.diags.shape[-2] != len(self.offsets):
            raise ValueError(
                'Argument `diags` of `SparseDIADataArray` must be of shape '
                f'(..., len(offsets), prod(dims)), but got {self.diags.shape}'
            )

        # check that diagonals contain zeros outside the bounds of the matrix using
        # equinox runtime checks
        error = (
            'Diagonals of a `SparseDIADataArray` must contain zeros outside the '
            'matrix bounds.'
        )
        for i, offset in enumerate(self.offsets):
            zero_slice = slice(None, offset) if offset >= 0 else slice(offset, None)
            check = self.diags[..., i, zero_slice] != 0
            eqx.error_if(self.diags, check, error)

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def layout(self) -> Layout:
        return dia

    @property
    def shape(self) -> tuple[int, ...]:
        return shape_sparsedia(self.diags)

    @property
    def mT(self) -> DataArray:
        offsets, diags = transpose_sparsedia(self.offsets, self.diags)
        return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]

    @property
    def ndiags(self) -> int:
        return len(self.offsets)

    def conj(self) -> DataArray:
        diags = self.diags.conj()
        return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]

    def _reshape_unchecked(self, *shape: int) -> DataArray:
        offsets, diags = reshape_sparsedia(self.offsets, self.diags, shape)
        return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]

    def broadcast_to(self, *shape: int) -> DataArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because the last two dimensions do '
                f'not match current shape dimensions, {self.shape}.'
            )

        offsets, diags = broadcast_sparsedia(self.offsets, self.diags, shape)
        return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]

    def powm(self, n: int) -> DataArray:
        offsets, diags = powm_sparsedia(self.offsets, self.diags, n)
        return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]

    def expm(self, *, max_squarings: int = 16) -> DataArray:
        warnings.warn(
            'A `SparseDIADataArray` has been converted to a `DenseDataArray` while '
            'computing its matrix exponential.',
            stacklevel=2,
        )
        x = sparsedia_to_array(self.offsets, self.diags)
        expm_x = jax.scipy.linalg.expm(x, max_squarings=max_squarings)
        return DenseDataArray(expm_x)

    def norm(self, *, psd: bool = False) -> Array:
        if psd:
            return self.trace()

        return self.asdense().norm(psd=psd)

    def trace(self) -> Array:
        return trace_sparsedia(self.offsets, self.diags)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        # return array if last two dimensions are modified, DataArray otherwise
        if in_last_two_dims(axis, self.ndim):
            if include_last_two_dims(axis, self.ndim):
                return self.diags.sum(axis)
            else:
                return self.to_jax().sum(axis)
        else:
            diags = self.diags.sum(axis)
            return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DataArray | Array:
        # return array if last two dimensions are modified, DataArray otherwise
        if in_last_two_dims(axis, self.ndim):
            if include_last_two_dims(axis, self.ndim):
                return self.diags.squeeze(axis)
            else:
                return self.to_jax().squeeze(axis)
        else:
            diags = self.diags.squeeze(axis)
            return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]

    def _eig(self) -> tuple[Array, DataArray]:
        warnings.warn(
            'A `SparseDIADataArray` has been converted to a `DenseDataArray` while '
            'attempting to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eig()

    def _eigh(self) -> tuple[Array, Array]:
        warnings.warn(
            'A `SparseDIADataArray` has been converted to a `DenseDataArray` while '
            'attempting to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigh()

    def _eigvals(self) -> Array:
        warnings.warn(
            'A `SparseDIADataArray` has been converted to a `DenseDataArray` while '
            'attempting to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvals()

    def _eigvalsh(self) -> Array:
        warnings.warn(
            'A `SparseDIADataArray` has been converted to a `DenseDataArray` while '
            'attempting to compute its eigen-decomposition.',
            stacklevel=2,
        )
        return self.asdense()._eigvalsh()

    def devices(self) -> set[jax.Device]:
        raise NotImplementedError

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return self.asdense().isherm(rtol=rtol, atol=atol)

    def to_jax(self) -> Array:
        return sparsedia_to_array(self.offsets, self.diags)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.asdense().__array__(dtype=dtype, copy=copy)

    def asdense(self) -> DenseDataArray:
        data = sparsedia_to_array(self.offsets, self.diags)
        return DenseDataArray(data)

    def assparsedia(
        self,
        offsets: tuple[int, ...] | None = None,  # noqa: ARG002
    ) -> SparseDIADataArray:
        return self

    def block_until_ready(self) -> DataArray:
        _ = self.diags.block_until_ready()
        return self

    def _repr_extra(self) -> str:
        # === array representation with dots instead of zeros
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            # match '0. +0.j' with any number of spaces
            pattern = r'(?<!\d)0\.(?:0+)?(?:e[+-]0+)?\s*[+-]\s*0\.(?:0+)?(?:e[+-]0+)?j'
        elif jnp.issubdtype(self.dtype, jnp.floating):
            # match '0.' with any number of spaces
            pattern = r'(?<!\d)0\.(?:0+)?(?:e[+-]0+)?\s*'
        elif jnp.issubdtype(self.dtype, jnp.integer):
            # match '0' with any number of spaces
            pattern = r'(?<!\d)0\s*'
        else:
            raise ValueError(
                'Unsupported dtype for `SparseDIADataArray` representation, got '
                f'{self.dtype}.'
            )

        # replace with a centered dot of the same length as the matched string
        replace_with_dot = lambda match: f'{"⋅":^{len(match.group(0))}}'
        data_str = re.sub(pattern, replace_with_dot, str(self.to_jax()))
        return f', ndiags={self.ndiags}\n{data_str}'

    def __mul__(self, y: DataArray | ArrayLike) -> DataArray:
        if isinstance(y, SparseDIADataArray):
            offsets, diags = mul_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]
        elif isinstance(y, DenseDataArray):
            offsets, diags = mul_sparsedia_array(self.offsets, self.diags, y.data)
            return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]
        elif isinstance(y, get_args(ArrayLike)):
            y_arr = jnp.asarray(y)
            if (
                y_arr.ndim == 0
                or (y_arr.ndim == 1 and y_arr.shape[0] == 1)
                or (y_arr.ndim >= 2 and y_arr.shape[-2:] == (1, 1))
            ):
                # scalar or batched scalar: broadcast onto diags directly
                diags = y_arr * self.diags
                return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]
            else:
                # full matrix: extract matching diagonals and multiply
                offsets, diags = mul_sparsedia_array(self.offsets, self.diags, y_arr)
                return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]

        return NotImplemented

    def __add__(self, y: DataArray | ArrayLike) -> DataArray:
        if isinstance(y, int | float) and y == 0:
            return self

        if isinstance(y, SparseDIADataArray):
            offsets, diags = add_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]
        elif isinstance(y, DenseDataArray):
            warnings.warn(
                'A sparse data array has been converted to dense layout due to '
                'element-wise addition with a dense data array.',
                stacklevel=2,
            )
            return self.asdense() + y
        elif isinstance(y, get_args(ArrayLike)):
            warnings.warn(
                'A sparse data array has been converted to dense layout due to '
                'element-wise addition with a raw array.',
                stacklevel=2,
            )
            return self.asdense() + DenseDataArray(jnp.asarray(y))

        return NotImplemented

    def __matmul__(self, y: DataArray | ArrayLike) -> DataArray:
        if isinstance(y, SparseDIADataArray):
            offsets, diags = matmul_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]
        elif isinstance(y, DenseDataArray):
            data = matmul_sparsedia_array(self.offsets, self.diags, y.data)
            return DenseDataArray(data)
        elif isinstance(y, get_args(ArrayLike)):
            data = matmul_sparsedia_array(self.offsets, self.diags, jnp.asarray(y))
            return DenseDataArray(data)

        return NotImplemented

    def __rmatmul__(self, y: DataArray | ArrayLike) -> DataArray:
        if isinstance(y, DenseDataArray):
            data = matmul_array_sparsedia(y.data, self.offsets, self.diags)
            return DenseDataArray(data)
        elif isinstance(y, get_args(ArrayLike)):
            data = matmul_array_sparsedia(jnp.asarray(y), self.offsets, self.diags)
            return DenseDataArray(data)

        return NotImplemented

    def __and__(self, y: DataArray) -> DataArray:
        if isinstance(y, SparseDIADataArray):
            offsets, diags = and_sparsedia_sparsedia(
                self.offsets, self.diags, y.offsets, y.diags
            )
            return replace(self, offsets=offsets, diags=diags)  # ty: ignore[invalid-argument-type]
        elif isinstance(y, DenseDataArray):
            return self.asdense() & y

        return NotImplemented

    def __rand__(self, y: DataArray) -> DataArray:
        if isinstance(y, DenseDataArray):
            return y & self.asdense()

        return NotImplemented

    def __pow__(self, power: int | _Metaω) -> DataArray:
        # to deal with the x**ω notation from equinox (used in diffrax internals)
        if isinstance(power, _Metaω):
            return _Metaω.__rpow__(power, self)

        diags = self.diags**power
        return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]

    def __getitem__(self, key: IndexType) -> DataArray:
        if key in (slice(None, None, None), Ellipsis):
            return self

        _check_key_in_batch_dims(key, self.ndim)
        diags = self.diags[key]
        return replace(self, diags=diags)  # ty: ignore[invalid-argument-type]


def _check_key_in_batch_dims(key: IndexType, ndim: int):
    full_slice = slice(None, None, None)
    valid_key = False
    if isinstance(key, int | slice):
        valid_key = ndim > 2
    if isinstance(key, Array):
        valid_key = key.ndim == 0 and ndim > 2
    elif isinstance(key, tuple):
        if Ellipsis in key:
            ellipsis_key = key.index(Ellipsis)
            key = (
                key[:ellipsis_key]
                + (full_slice,) * (ndim - len(key) + 1)
                + key[ellipsis_key + 1 :]
            )

        valid_key = (
            len(key) <= ndim - 2
            or (len(key) == ndim - 1 and key[-1] == full_slice)
            or (len(key) == ndim and key[-2] == full_slice and key[-1] == full_slice)
        )

    if not valid_key:
        raise NotImplementedError(
            'Getting items from non batching dimensions of a `SparseDIADataArray` is '
            'not supported.'
        )
