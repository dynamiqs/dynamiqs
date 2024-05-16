from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from .qarray import QArray


class SparseQArray(QArray):
    diags: Array
    offsets: tuple[int, ...] = eqx.field(static=True)

    def __add__(
        self, other: ScalarLike | ArrayLike | SparseQArray
    ) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                return self
            else:
                return self.to_dense() + other
        elif isinstance(other, ArrayLike):
            return self.to_dense() + other
        elif isinstance(other, SparseQArray):
            return self._add_dia(other)

        return NotImplemented

    def __radd__(self, other: Array) -> Array:
        return self + other

    def _add_dia(self, other: SparseQArray) -> SparseQArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))

        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] += other_diag
            else:
                out_offsets_diags[other_offset] = other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.array([out_offsets_diags[offset] for offset in out_offsets])

        return SparseQArray(out_diags, out_offsets)

    def __sub__(
        self, other: ScalarLike | ArrayLike | SparseQArray
    ) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                return self
            else:
                return self.to_dense() - other
        elif isinstance(other, ArrayLike):
            return self.to_dense() - other
        elif isinstance(other, SparseQArray):
            return self._sub_dia(other)

        return NotImplemented

    def __rsub__(self, other: Array) -> Array:
        return -self + other

    def _sub_dia(self, other: SparseQArray) -> SparseQArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))

        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] -= other_diag
            else:
                out_offsets_diags[other_offset] = -other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.array([out_offsets_diags[offset] for offset in out_offsets])

        return SparseQArray(out_diags, out_offsets)

    def __mul__(self, other: Array | SparseQArray) -> Array | SparseQArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                empty_diags = jnp.empty(0, self.shape[-1])
                return SparseQArray(empty_diags, ())
            return SparseQArray(other * self.diags, self.offsets)
        elif isinstance(other, Array):
            return self._mul_dense(other)
        elif isinstance(other, SparseQArray):
            return self._mul_dia(other)

        return NotImplemented

    def __rmul__(self, other: ArrayLike) -> Array:
        return self * other

    def _mul_dense(self, other: Array) -> SparseQArray:
        out_diags = jnp.zeros_like(self.diags)
        N = other.shape[0]

        for i, (self_offset, self_diag) in enumerate(zip(self.offsets, self.diags)):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            other_diag = jnp.diagonal(other, self_offset)
            out_diags = out_diags.at[i, start:end].set(
                other_diag * self_diag[start:end]
            )

        return SparseQArray(out_diags, self.offsets)

    def _mul_dia(self, other: SparseQArray) -> SparseQArray:
        out_diags, out_offsets = [], []

        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                if self_offset != other_offset:
                    continue
                out_diags.append(self_diag * other_diag)
                out_offsets.append(other_offset)

        return SparseQArray(jnp.stack(out_diags), tuple(out_offsets))
