from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from .qarray import QArray

__all__ = ['SparseQArray', 'to_dense', 'to_sparse']


class SparseQArray(QArray):
    diags: Array
    offsets: tuple[int, ...] = eqx.field(static=True)
    dims: tuple[int, ...] = eqx.field(static=True)

    def to_dense(self) -> Array:
        return to_dense(self)

    def __add__(
        self, other: ScalarLike | ArrayLike | SparseQArray
    ) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                return self
            return self.to_dense() + other
        elif isinstance(other, ArrayLike):
            return self.to_dense() + other
        elif isinstance(other, SparseQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._add_sparse(other)

        return NotImplemented

    def _add_sparse(self, other: SparseQArray) -> SparseQArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))
        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] += other_diag
            else:
                out_offsets_diags[other_offset] = other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.stack([out_offsets_diags[offset] for offset in out_offsets])

        return SparseQArray(out_diags, out_offsets, self.dims)

    def __radd__(self, other: Array) -> Array:
        return self + other

    def __sub__(
        self, other: ScalarLike | ArrayLike | SparseQArray
    ) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                return self
            return self.to_dense() - other
        elif isinstance(other, ArrayLike):
            return self.to_dense() - other
        elif isinstance(other, SparseQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._sub_sparse(other)

        return NotImplemented

    def _sub_sparse(self, other: SparseQArray) -> SparseQArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))
        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] -= other_diag
            else:
                out_offsets_diags[other_offset] = -other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.array([out_offsets_diags[offset] for offset in out_offsets])

        return SparseQArray(out_diags, out_offsets, self.dims)

    def __rsub__(self, other: Array) -> Array:
        return -self + other

    def __mul__(self, other: Array | SparseQArray) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                diags = jnp.empty(0, self.shape[-1])
                offsets = ()
                return SparseQArray(diags, offsets, self.dims)
            return SparseQArray(other * self.diags, self.offsets, self.dims)
        elif isinstance(other, Array):
            return self._mul_dense(other)
        elif isinstance(other, SparseQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._mul_sparse(other)

        return NotImplemented

    def _mul_dense(self, other: Array) -> SparseQArray:
        N = other.shape[0]
        out_diags = jnp.zeros_like(self.diags)
        for i, (self_offset, self_diag) in enumerate(zip(self.offsets, self.diags)):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            other_diag = jnp.diagonal(other, self_offset)
            out_diags = out_diags.at[i, start:end].set(
                other_diag * self_diag[start:end]
            )

        return SparseQArray(out_diags, self.offsets, self.dims)

    def _mul_sparse(self, other: SparseQArray) -> SparseQArray:
        out_diags, out_offsets = [], []
        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                if self_offset != other_offset:
                    continue
                out_diags.append(self_diag * other_diag)
                out_offsets.append(other_offset)

        return SparseQArray(jnp.stack(out_diags), tuple(out_offsets), self.dims)

    def __rmul__(self, other: ArrayLike) -> Array:
        return self * other


def _check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'QArrays have incompatible dimensions. Got {dims1} and {dims2}.'
        )


def to_dense(sparse: SparseQArray) -> Array:
    r"""Returns the input matrix in the Dense format.

    Parameters:
        sparse: A sparse matrix, containing diagonals and their offsets.

    Returns:
        Array: A dense matrix representation of the input sparse matrix.
    """
    N = sparse.dims[0]
    out = jnp.zeros((N, N))
    for offset, diag in zip(sparse.offsets, sparse.diags):
        start = max(0, offset)
        end = min(N, N + offset)
        out += jnp.diag(diag[start:end], k=offset)
    return out


def to_sparse(other: ArrayLike) -> SparseQArray:
    r"""Returns the input matrix in the SparseDIA format.

    This should be used when a user wants to turn a dense matrix that
    presents sparse features in the SparseDIA custom format.

    Args:
        other: NxN matrix to turn from dense to SparseDIA format.

    Returns:
        SparseDIA object which has 2 main attributes:
            object.diags: Array where each row is a diagonal.
            object.offsets: tuple of integers that represents the
                            respective offsets of the diagonals
    """
    diagonals = []
    offsets = []

    N = other.shape[0]
    offset_range = 2 * N - 1
    offset_center = offset_range // 2

    if not isinstance(other, Array):
        other = jnp.asarray(other.array)

    for offset in range(-offset_center, offset_center + 1):
        diagonal = jnp.diagonal(other, offset=offset)
        if jnp.any(diagonal != 0):
            if offset > 0:
                padding = (offset, 0)
            elif offset < 0:
                padding = (0, -offset)
            else:
                padding = (0, 0)

            padded_diagonal = jnp.pad(
                diagonal, padding, mode='constant', constant_values=0
            )
            diagonals.append(padded_diagonal)
            offsets.append(offset)

    diagonals = jnp.array(diagonals)
    offsets = tuple(offsets)

    return SparseQArray(diagonals, offsets, (N, N))
