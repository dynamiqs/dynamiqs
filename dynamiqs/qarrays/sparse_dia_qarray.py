from __future__ import annotations

import functools
import warnings
from collections import defaultdict
from typing import get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import concrete_or_error
from jaxtyping import Array, ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar
from .dense_qarray import DenseQArray
from .types import QArray, QArrayLike, asqarray

__all__ = ['SparseDIAQArray', 'to_dense', 'to_sparse_dia']


class SparseDIAQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array

    def __init__(self, dims: tuple[int, ...], offsets: tuple[int, ...], diags: Array):
        super().__init__(dims)
        self.offsets, self.diags = _compress_dia(offsets, jnp.asarray(diags))

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

    def conj(self) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags.conj())

    def reshape(self, *shape: int) -> QArray:
        raise NotImplementedError

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

    def ptrace(self, keep: tuple[int, ...]) -> QArray:
        raise NotImplementedError

    def powm(self):
        raise NotImplementedError

    def expm(self):
        raise NotImplementedError

    def norm(self) -> Array:
        return self.trace()

    def trace(self) -> Array:
        main_diag_mask = np.asarray(self.offsets) == 0
        if np.any(main_diag_mask):
            return jnp.sum(self.diags[..., main_diag_mask, :], axis=(-1, -2))
        else:
            return jnp.zeros(self.shape[:-2])

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Array:
        raise NotImplementedError

    def squeeze(self):
        raise NotImplementedError

    def _eigh(self) -> tuple[Array, Array]:
        raise NotImplementedError

    def _eigvals(self) -> Array:
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        raise NotImplementedError

    def devices(self) -> set[jax.Device]:
        raise NotImplementedError

    def to_dense(self) -> DenseQArray:
        return to_dense(self)

    def isherm(self) -> bool:
        raise NotImplementedError

    def to_qutip(self) -> Qobj:
        raise NotImplementedError

    def to_jax(self) -> Array:
        return self.to_dense().to_jax()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        raise self.to_dense().__array__(dtype=dtype, copy=copy)

    def __mul__(self, other: QArrayLike) -> QArray:
        super().__mul__(other)

        if _is_batched_scalar(other):
            diags, offsets = other * self.diags, self.offsets
            return SparseDIAQArray(offsets=offsets, diags=diags, dims=self.dims)
        elif isinstance(other, SparseDIAQArray):
            return self._mul_sparse(other)
        elif isinstance(other, get_args(QArrayLike)):
            return self._mul_dense(other)

        return NotImplemented

    def _mul_dense(self, other: QArrayLike) -> QArray:
        # initialize the output diagonals
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        out_shape = (*batch_shape, len(self.offsets), self.shape[-1])
        out_diags = jnp.zeros(out_shape, dtype=self.dtype)

        # loop over each diagonal of the sparse matrix and fill the output
        for i, self_offset in enumerate(self.offsets):
            self_slice = _dia_slice(self_offset)
            other_diag = jnp.diagonal(other, offset=self_offset, axis1=-2, axis2=-1)
            out_diag = other_diag * self.diags[..., i, self_slice]
            out_diags = out_diags.at[..., i, self_slice].set(out_diag)

        return SparseDIAQArray(self.dims, self.offsets, out_diags)

    def _mul_sparse(self, other: SparseDIAQArray) -> QArray:
        # compute the output offsets as the intersection of offsets
        out_offsets, self_ind, other_ind = np.intersect1d(
            self.offsets, other.offsets, return_indices=True
        )

        # initialize the output diagonals
        batch_shape = jnp.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        out_shape = (*batch_shape, len(out_offsets), self.shape[-1])
        out_diags = jnp.zeros(out_shape, dtype=self.dtype)

        # loop over each output offset and fill the output
        for i, offset in enumerate(out_offsets):
            self_diag = self.diags[..., self.offsets.index(offset), :]
            other_diag = other.diags[..., other.offsets.index(offset), :]
            out_diag = self_diag * other_diag
            out_diags = out_diags.at[..., i, _dia_slice(offset)].set(out_diag)

        return SparseDIAQArray(self.dims, tuple(out_offsets), out_diags)

    def __truediv__(self, other: QArrayLike) -> QArray:
        raise NotImplementedError

    def __add__(self, other: QArrayLike) -> QArray:
        if _is_batched_scalar(other):
            if other == 0:
                return self
            warnings.warn(
                '`to_dense` was called on a SparseDIAQArray due to addition with a '
                'scalar. The array is no longer in sparse format.',
                stacklevel=2,
            )
            return self.to_dense() + other
        elif isinstance(other, ArrayLike):
            warnings.warn(
                '`to_dense` was called on a SparseDIAQArray due to addition with a '
                'dense array. The array is no longer in sparse format.',
                stacklevel=2,
            )
            return self.to_dense() + other
        elif isinstance(other, SparseDIAQArray):
            return self._add_sparse(other)
        elif isinstance(other, DenseQArray):
            return self.to_dense() + other

        return NotImplemented

    def _add_sparse(self, other: SparseDIAQArray) -> QArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))
        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] += other_diag
            else:
                out_offsets_diags[other_offset] = other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.stack([out_offsets_diags[offset] for offset in out_offsets])

        return SparseDIAQArray(self.dims, out_offsets, out_diags)

    def __matmul__(self, other: QArrayLike) -> QArray:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=True, other=other)
        elif isinstance(other, SparseDIAQArray):
            return self._matmul_dia(other=other)

        return NotImplemented

    def __rmatmul__(self, other: QArrayLike) -> QArray:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=False, other=other)

        return NotImplemented

    def _matmul_dense(self, left_matmul: bool, other: QArrayLike) -> QArray:
        N = other.shape[0]
        out = jnp.zeros_like(other)
        for self_offset, self_diag in zip(self.offsets, self.diags):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            top = max(0, -self_offset)
            bottom = top + end - start

            if left_matmul:
                out = out.at[top:bottom, :].add(
                    self_diag[start:end, None] * other[start:end, :]
                )
            else:
                out = out.at[:, start:end].add(
                    self_diag[start:end, None].T * other[:, top:bottom]
                )

        return DenseQArray(self.dims, out)

    def _matmul_dia(self, other: SparseDIAQArray) -> QArray:
        N = other.diags.shape[1]
        diag_dict = defaultdict(lambda: jnp.zeros(N))

        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                result_offset = self_offset + other_offset

                if abs(result_offset) > N - 1:
                    continue

                sA, sB = max(0, -other_offset), max(0, other_offset)
                eA, eB = min(N, N - other_offset), min(N, N + other_offset)

                diag_dict[result_offset] = (
                    diag_dict[result_offset]
                    .at[sB:eB]
                    .add(self_diag[sA:eA] * other_diag[sB:eB])
                )

        out_offsets = sorted(diag_dict.keys())
        out_diags = [diag_dict[offset] for offset in out_offsets]

        return SparseDIAQArray(self.dims, tuple(out_offsets), jnp.vstack(out_diags))

    def _kronecker_dia(self, other: SparseDIAQArray) -> SparseDIAQArray:
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

    def __and__(self, other: QArrayLike) -> QArray:
        if isinstance(other, SparseDIAQArray):
            return self._kronecker_dia(other)
        elif isinstance(other, get_args(QArrayLike)):
            return self.to_dense() & asqarray(other)

        return NotImplemented

    def _pow(self, power: int) -> QArray:  # noqa: ARG002
        return NotImplemented

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
    # For exemple, a diagonal with offset 2 is stored as [0, 0, a, b, ..., z], and
    # _dia_slice(2) will return the slice(2, None) to select [a, b, ..., z].
    return slice(offset, None) if offset >= 0 else slice(None, offset)


def _compress_dia(offsets: tuple[int, ...], diags: ArrayLike) -> SparseDIAQArray:
    # compute unique offsets
    out_offsets, inverse_ind = np.unique(offsets, return_inverse=True)

    # initialize output diagonals
    diags_shape = (*diags.shape[:-2], len(out_offsets), diags.shape[-1])
    out_diags = jnp.zeros(diags_shape, dtype=diags.dtype)

    # loop over each offset and fill the output
    for i in range(len(out_offsets)):
        mask = inverse_ind == i
        diag = jnp.sum(diags[..., mask, :], axis=-2)
        out_diags = out_diags.at[..., i, :].set(diag)

    return tuple(out_offsets), out_diags


def to_dense(x: SparseDIAQArray) -> DenseQArray:
    r"""Convert a sparse `QArray` into a dense `Qarray`.

    Args:
        x: A sparse matrix, containing diagonals and their offsets.

    Returns:
        Array: A dense matrix representation of the input sparse matrix.
    """
    out = jnp.zeros(x.shape, dtype=x.dtype)
    for offset, diag in zip(x.offsets, x.diags):
        out += _vectorized_diag(diag, offset)
    return DenseQArray(x.dims, out)


@functools.partial(jnp.vectorize, signature='(n)->(n,n)', excluded={1})
def _vectorized_diag(diag: Array, offset: int) -> Array:
    return jnp.diag(diag[_dia_slice(offset)], k=offset)


def to_sparse_dia(x: QArrayLike) -> SparseDIAQArray:
    r"""Convert the input to a sparse DIA qarray.

    Args:
        x: Input data, in any qarray-like format.

    Returns:
        The sparse DIA matrix representation of the input matrix.
    """
    if isinstance(x, SparseDIAQArray):
        return x
    elif isinstance(x, DenseQArray):
        dims = x.dims
        x = x.to_jax()
    elif isinstance(x, get_args(ArrayLike)):
        dims = (x.shape[-1],)
        x = jnp.asarray(x)
    else:
        raise TypeError('Input must be a `QArrayLike` object.')

    concrete_or_error(None, x, '`to_sparse_dia` does not support tracing.')
    offsets = _find_offsets(x)
    diags = _construct_diags(offsets, x)
    return SparseDIAQArray(dims=dims, offsets=offsets, diags=diags)


def _find_offsets(x: Array) -> tuple[int, ...]:
    indices = np.nonzero(x)
    return tuple(np.unique(indices[1] - indices[0]))


@functools.partial(jax.jit, static_argnums=(0,))
def _construct_diags(offsets: tuple[int, ...], x: Array) -> Array:
    n = x.shape[0]
    diags = jnp.zeros((*x.shape[:-2], len(offsets), n), dtype=x.dtype)

    for i, offset in enumerate(offsets):
        diagonal = jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)
        diags = diags.at[..., i, _dia_slice(offset)].set(diagonal)

    return diags
