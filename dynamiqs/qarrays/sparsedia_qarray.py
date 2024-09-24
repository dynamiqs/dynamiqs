from __future__ import annotations

import functools
import re
import warnings
from collections import defaultdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import concrete_or_error
from jaxtyping import Array, ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar, cdtype
from .dense_qarray import DenseQArray, _getjaxarray
from .layout import Layout, dia
from .qarray import (
    QArray,
    QArrayLike,
    _in_last_two_dims,
    _include_last_two_dims,
    isqarraylike,
)

__all__ = ['SparseDIAQArray']


def _sparsedia_to_dense(x: SparseDIAQArray) -> DenseQArray:
    out = jnp.zeros(x.shape, dtype=cdtype())
    for i, offset in enumerate(x.offsets):
        out += _vectorized_diag(x.diags[..., i, :], offset)
    return DenseQArray(x.dims, out)


@functools.partial(jnp.vectorize, signature='(n)->(n,n)', excluded={1})
def _vectorized_diag(diag: Array, offset: int) -> Array:
    return jnp.diag(diag[_dia_slice(offset)], k=offset)


def _array_to_sparsedia(x: Array, dims: tuple[int, ...]) -> SparseDIAQArray:
    concrete_or_error(None, x, '`_array_to_sparsedia` does not support tracing.')
    offsets = _find_offsets(x)
    diags = _construct_diags(offsets, x)
    return SparseDIAQArray(dims=dims, offsets=offsets, diags=diags)


def _find_offsets(x: Array) -> tuple[int, ...]:
    indices = np.nonzero(x)
    return tuple(np.unique(indices[-1] - indices[-2]))


@functools.partial(jax.jit, static_argnums=(0,))
def _construct_diags(offsets: tuple[int, ...], x: Array) -> Array:
    n = x.shape[-1]
    diags = jnp.zeros((*x.shape[:-2], len(offsets), n), dtype=cdtype())

    for i, offset in enumerate(offsets):
        diagonal = jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)
        diags = diags.at[..., i, _dia_slice(offset)].set(diagonal)

    return diags


def _sparsedia_to_qobj(x: SparseDIAQArray) -> Qobj | list[Qobj]:
    return x.to_dense().asqobj()


class SparseDIAQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        # check diags and offsets have the right type and shape before compressing them
        if not isinstance(self.offsets, tuple):
            raise TypeError(
                'Argument `offsets` of `SparseDIAQArray` must be a tuple but got '
                f'{type(self.offsets)}'
            )

        if self.diags.ndim < 2 or self.diags.shape[-2] != len(self.offsets):
            raise ValueError(
                'Argument `diags` of `SparseDIAQArray` must be of shape '
                f'(..., len(offsets), prod(dims)), but got {self.diags.shape}'
            )

        # if the code is jitted, disable following checks
        if isinstance(self.diags, jax.core.Tracer):
            return

        # check that diagonals contain zeros outside the bounds of the matrix
        for i, offset in enumerate(self.offsets):
            if (offset < 0 and jnp.any(self.diags[..., i, offset:] != 0)) or (
                offset > 0 and jnp.any(self.diags[..., i, :offset] != 0)
            ):
                raise ValueError(
                    'Diagonals of a `SparseDIAQArray` must contain zeros outside the '
                    'matrix bounds.'
                )

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def layout(self) -> Layout:
        return dia

    @property
    def shape(self) -> tuple[int, ...]:
        N = self.diags.shape[-1]
        return (*self.diags.shape[:-2], N, N)

    @property
    def mT(self) -> QArray:
        # initialize the output diagonals
        out_diags = jnp.zeros_like(self.diags)

        # compute output offsets
        out_offsets = tuple(-x for x in self.offsets)

        # loop over each offset and fill the output
        for i, self_offset in enumerate(self.offsets):
            self_slice = _dia_slice(self_offset)
            out_slice = _dia_slice(-self_offset)
            out_diags = out_diags.at[..., i, out_slice].set(
                self.diags[..., i, self_slice]
            )

        return SparseDIAQArray(self.dims, out_offsets, out_diags)

    @property
    def ndiags(self) -> int:
        return len(self.offsets)

    def conj(self) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags.conj())

    def reshape(self, *shape: int) -> QArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot reshape to shape {shape} because'
                f' the last two dimensions do not match current '
                f'shape dimensions ({self.shape})'
            )

        shape = (*shape[:-2], len(self.offsets), self.diags.shape[-1])
        diags = jnp.reshape(self.diags, shape)
        return SparseDIAQArray(diags=diags, offsets=self.offsets, dims=self.dims)

    def broadcast_to(self, *shape: int) -> QArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because'
                f' the last two dimensions do not match current '
                f'shape dimensions ({self.shape})'
            )

        shape = (*shape[:-2], len(self.offsets), self.diags.shape[-1])
        diags = jnp.broadcast_to(self.diags, shape)
        return SparseDIAQArray(diags=diags, offsets=self.offsets, dims=self.dims)

    def ptrace(self, *keep: int) -> QArray:
        raise NotImplementedError

    def powm(self, n: int) -> QArray:
        if n == 0:
            eye = jnp.eye(self.shape[-1], dtype=self.dtype)
            batched_eye = jnp.broadcast_to(eye, self.shape)
            return SparseDIAQArray(self.dims, self.offsets, batched_eye)
        if n == 1:
            return self
        else:
            return self @ self.powm(n - 1)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # todo: implement dia specific method or raise warning for dense conversion
        return _sparsedia_to_dense(self).expm(max_squarings=max_squarings)

    def norm(self) -> Array:
        return self.trace()

    def trace(self) -> Array:
        main_diag_mask = np.asarray(self.offsets) == 0
        if np.any(main_diag_mask):
            return jnp.sum(self.diags[..., main_diag_mask, :], axis=(-1, -2))
        else:
            return jnp.zeros(self.shape[:-2])

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.sum(axis)
            else:
                return self.asjaxarray().sum(axis)
        else:
            return SparseDIAQArray(self.dims, self.offsets, self.diags.sum(axis))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.squeeze(axis)
            else:
                return self.asjaxarray().squeeze(axis)
        else:
            return SparseDIAQArray(self.dims, self.offsets, self.diags.squeeze(axis))

    def _eigh(self) -> tuple[Array, Array]:
        raise NotImplementedError

    def _eigvals(self) -> Array:
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        raise NotImplementedError

    def devices(self) -> set[jax.Device]:
        raise NotImplementedError

    def to_dense(self) -> DenseQArray:
        return _sparsedia_to_dense(self)

    def isherm(self) -> bool:
        raise NotImplementedError

    def asqobj(self) -> Qobj | list[Qobj]:
        return _sparsedia_to_qobj(self)

    def asjaxarray(self) -> Array:
        return self.to_dense().asjaxarray()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.to_dense().__array__(dtype=dtype, copy=copy)

    def __repr__(self) -> str:
        # === array representation with dots instead of zeros
        # match '0. +0.j' with any number of spaces
        pattern = r'0\.\s*\+0\.j'
        # replace with a centered dot of the same length as the matched string
        replace_with_dot = lambda match: f"{'⋅':^{len(match.group(0))}}"
        data_str = re.sub(pattern, replace_with_dot, str(self.asjaxarray()))
        return super().__repr__() + f', ndiags={self.ndiags}\n{data_str}'

    def __mul__(self, other: QArrayLike) -> QArray:
        super().__mul__(other)

        if _is_batched_scalar(other):
            return self._mul_scalar(other)
        elif isinstance(other, SparseDIAQArray):
            return self._mul_sparse(other)
        elif isqarraylike(other):
            other = _getjaxarray(other)
            return self._mul_dense(other)

        return NotImplemented

    def _mul_scalar(self, other: QArrayLike) -> QArray:
        diags, offsets = other * self.diags, self.offsets
        return SparseDIAQArray(self.dims, offsets, diags)

    def _mul_sparse(self, other: SparseDIAQArray) -> QArray:
        # we check that the offsets are unique, as they should with the class __init__
        assert len(set(self.offsets)) == len(self.offsets)
        assert len(set(other.offsets)) == len(other.offsets)

        # compute the output offsets as the intersection of offsets
        out_offsets, self_ind, other_ind = np.intersect1d(
            self.offsets, other.offsets, assume_unique=True, return_indices=True
        )

        # initialize the output diagonals
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        out_shape = (*batch_shape, len(out_offsets), self.shape[-1])
        out_diags = jnp.zeros(out_shape, dtype=cdtype())

        # loop over each output offset and fill the output
        for i in range(len(out_offsets)):
            self_diag = self.diags[..., self_ind[i], :]
            other_diag = other.diags[..., other_ind[i], :]
            out_diag = self_diag * other_diag
            out_diags = out_diags.at[..., i, :].set(out_diag)

        return SparseDIAQArray(self.dims, tuple(out_offsets), out_diags)

    def _mul_dense(self, other: Array) -> QArray:
        # initialize the output diagonals
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        out_shape = (*batch_shape, len(self.offsets), self.shape[-1])
        out_diags = jnp.zeros(out_shape, dtype=cdtype())

        # loop over each diagonal of the sparse matrix and fill the output
        for i, self_offset in enumerate(self.offsets):
            self_slice = _dia_slice(self_offset)
            other_diag = jnp.diagonal(other, offset=self_offset, axis1=-2, axis2=-1)
            out_diag = other_diag * self.diags[..., i, self_slice]
            out_diags = out_diags.at[..., i, self_slice].set(out_diag)

        return SparseDIAQArray(self.dims, self.offsets, out_diags)

    def __truediv__(self, other: QArrayLike) -> QArray:
        raise NotImplementedError

    def __add__(self, other: QArrayLike) -> QArray:
        super().__add__(other)

        warning_dense_addition = (
            'A sparse array has been converted to dense format due to '
            'addition with a scalar or dense array.'
        )

        if _is_batched_scalar(other):
            warnings.warn(warning_dense_addition, stacklevel=2)
            return self._add_scalar(other)
        elif isinstance(other, SparseDIAQArray):
            return self._add_sparse(other)
        elif isqarraylike(other):
            warnings.warn(warning_dense_addition, stacklevel=2)
            other = _getjaxarray(other)
            return self._add_dense(other)

        return NotImplemented

    def _add_scalar(self, other: QArrayLike) -> QArray:
        return self.to_dense() + other

    def _add_sparse(self, other: SparseDIAQArray) -> QArray:
        # compute the output offsets
        out_offsets = np.union1d(self.offsets, other.offsets)

        # initialize the output diagonals
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        diags_shape = (*batch_shape, len(out_offsets), self.diags.shape[-1])
        out_diags = jnp.zeros(diags_shape, dtype=cdtype())

        # loop over each offset and fill the output
        for i, offset in enumerate(out_offsets):
            if offset in self.offsets:
                self_diag = self.diags[..., self.offsets.index(offset), :]
                out_diags = out_diags.at[..., i, :].add(self_diag)
            if offset in other.offsets:
                other_diag = other.diags[..., other.offsets.index(offset), :]
                out_diags = out_diags.at[..., i, :].add(other_diag)

        return SparseDIAQArray(self.dims, tuple(out_offsets), out_diags)

    def _add_dense(self, other: Array) -> QArray:
        return self.to_dense() + other

    def __matmul__(self, other: QArrayLike) -> QArray:
        if _is_batched_scalar(other):
            raise TypeError('Attempted matrix product between a scalar and a QArray.')

        if isinstance(other, SparseDIAQArray):
            return self._matmul_dia(other)
        elif isqarraylike(other):
            other = _getjaxarray(other)
            return self._matmul_dense(other, left_matmul=True)

        return NotImplemented

    def __rmatmul__(self, other: QArrayLike) -> QArray:
        if _is_batched_scalar(other):
            raise TypeError('Attempted matrix product between a scalar and a QArray.')

        if isqarraylike(other):
            other = _getjaxarray(other)
            return self._matmul_dense(other, left_matmul=False)

        return NotImplemented

    def _matmul_dia(self, other: SparseDIAQArray) -> QArray:
        N = other.diags.shape[-1]
        broadcast_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        diag_dict = defaultdict(
            lambda: jnp.zeros((*broadcast_shape, N), dtype=cdtype())
        )

        for i, self_offset in enumerate(self.offsets):
            self_diag = self.diags[..., i, :]
            for j, other_offset in enumerate(other.offsets):
                other_diag = other.diags[..., j, :]
                result_offset = self_offset + other_offset

                if abs(result_offset) > N - 1:
                    continue

                diag = (
                    self_diag[..., _dia_slice(-other_offset)]
                    * other_diag[..., _dia_slice(other_offset)]
                )
                diag_dict[result_offset] = (
                    diag_dict[result_offset].at[..., _dia_slice(other_offset)].add(diag)
                )

        out_offsets = sorted(diag_dict.keys())
        out_diags = [diag_dict[offset] for offset in out_offsets]

        out_diags = jnp.stack(out_diags)
        out_diags = jnp.moveaxis(out_diags, 0, -2)

        return SparseDIAQArray(self.dims, tuple(out_offsets), out_diags)

    def _matmul_dense(self, other: Array, left_matmul: bool) -> QArray:
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        out_shape = (*batch_shape, self.shape[-2], other.shape[-1])
        out = jnp.zeros(out_shape, dtype=cdtype())
        for i, self_offset in enumerate(self.offsets):
            self_diag = self.diags[..., i, :]
            slice_in = _dia_slice(self_offset)
            slice_out = _dia_slice(-self_offset)
            if left_matmul:
                out = out.at[..., slice_out, :].add(
                    self_diag[..., slice_in, None] * other[..., slice_in, :]
                )
            else:
                out = out.at[..., :, slice_in].add(
                    self_diag[..., slice_in, None].T * other[..., :, slice_out]
                )

        return DenseQArray(self.dims, out)

    def __and__(self, other: QArray) -> QArray:
        if isinstance(other, SparseDIAQArray):
            return self._and_dia(other)
        elif isinstance(other, DenseQArray):
            return self.to_dense() & other
        else:
            return NotImplemented

    def __rand__(self, other: QArray) -> QArray:
        if isinstance(other, DenseQArray):
            return other & self.to_dense()
        else:
            return NotImplemented

    def _and_dia(self, other: SparseDIAQArray) -> SparseDIAQArray:
        # compute new offsets
        N = other.diags.shape[-1]
        self_offsets = np.asarray(self.offsets)
        other_offsets = np.asarray(other.offsets)
        out_offsets = tuple(np.ravel(self_offsets[:, None] * N + other_offsets))

        # compute new diagonals and dimensions
        out_diags = jnp.kron(self.diags, other.diags)
        out_dims = self.dims + other.dims

        # merge duplicate offsets and return
        out_offsets, out_diags = _compress_dia(out_offsets, out_diags)
        return SparseDIAQArray(dims=out_dims, offsets=out_offsets, diags=out_diags)

    def _pow(self, power: int) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags**power)

    def __getitem__(self, key: int | slice) -> QArray:
        full = slice(None, None, None)

        if key in (full, Ellipsis):
            return self

        if isinstance(key, (int, slice)):
            is_key_valid = self.ndim > 2
        elif isinstance(key, tuple):
            if Ellipsis in key:
                ellipsis_key = key.index(Ellipsis)
                key = (
                    key[:ellipsis_key]
                    + (full,) * (self.ndim - len(key) + 1)
                    + key[ellipsis_key + 1 :]
                )

            is_key_valid = (
                len(key) <= self.ndim - 2
                or (len(key) == self.ndim - 1 and key[-1] == full)
                or (len(key) == self.ndim and key[-2] == full and key[-1] == full)
            )
        else:
            raise IndexError('Should never happen')

        if not is_key_valid:
            raise NotImplementedError(
                'Getting items for non batching dimensions of '
                'SparseDIA is not supported yet'
            )

        return SparseDIAQArray(
            diags=self.diags[key], offsets=self.offsets, dims=self.dims
        )


def _dia_slice(offset: int) -> slice:
    # Return the slice that selects the non-zero elements of a diagonal of given offset.
    # For example, a diagonal with offset 2 is stored as [0, 0, a, b, ..., z], and
    # _dia_slice(2) will return the slice(2, None) to select [a, b, ..., z].
    return slice(offset, None) if offset >= 0 else slice(None, offset)


def _compress_dia(offsets: tuple[int, ...], diags: ArrayLike) -> SparseDIAQArray:
    # compute unique offsets
    out_offsets, inverse_ind = np.unique(offsets, return_inverse=True)

    # initialize output diagonals
    diags_shape = (*diags.shape[:-2], len(out_offsets), diags.shape[-1])
    out_diags = jnp.zeros(diags_shape, dtype=cdtype())

    # loop over each offset and fill the output
    for i in range(len(out_offsets)):
        mask = inverse_ind == i
        diag = jnp.sum(diags[..., mask, :], axis=-2)
        out_diags = out_diags.at[..., i, :].set(diag)

    return tuple(out_offsets), out_diags
