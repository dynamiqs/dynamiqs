from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from .qarray import QArray, pack_dims


class SparseQArray(QArray):
    diags: Array
    offsets: tuple[int, ...] = eqx.field(static=True)
    dims: tuple[int, ...] = eqx.field(static=True)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.diags.shape

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    @pack_dims
    def mT(self) -> SparseQArray:
        N = self.dims[0]
        out_diags = jnp.zeros_like(self.diags)
        for i, (self_offset, self_diag) in enumerate(zip(self.offsets, self.diags)):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            out_diags = out_diags.at[i, start - self_offset : end - self_offset].set(
                self_diag[start:end]
            )
        return SparseQArray(out_diags, -1 * self.offsets, self.dims)

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
