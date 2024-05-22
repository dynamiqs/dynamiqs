from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from .qarray import QArray


class SparseQArray(QArray):
    diags: Array
    offsets: tuple[int, ...] = eqx.field(static=True)
    dims: tuple[int, ...] = eqx.field(static=True)

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

    def _matmul_dense(self, left_matmul: bool, other: ArrayLike) -> Array:
        N = other.shape[0]
        out = jnp.zeros_like(other)
        for self_offset, self_diag in zip(self.offsets, self.diags):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            top = max(0, -self_offset)
            bottom = top + end - start
            out = jax.lax.cond(
                left_matmul,
                lambda _,
                out=out,
                top=top,
                bottom=bottom,
                self_diag=self_diag,
                start=start,
                end=end: out.at[top:bottom, :].add(
                    self_diag[start:end, None] * other[start:end, :]
                ),
                lambda _,
                out=out,
                top=top,
                bottom=bottom,
                self_diag=self_diag,
                start=start,
                end=end: out.at[:, start:end].add(
                    jnp.expand_dims(self_diag[start:end], axis=0) * other[:, top:bottom]
                ),
                operand=None,
            )
        return out

    def _matmul_dia(self, other: SparseQArray) -> tuple[Array, list[int]]:
        N = other.diags.shape[-1]
        out_diags, out_offsets = [], []
        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                out_diag = jnp.zeros_like(self_diag)
                sA, sB = max(0, -other_offset), max(0, other_offset)
                eA, eB = min(N, N - other_offset), min(N, N + other_offset)
                out_diag = out_diag.at[sB:eB].add(self_diag[sA:eA] * other_diag[sB:eB])
                out_diags.append(out_diag)
                out_offsets.append(self_offset + other_offset)
        return SparseQArray(jnp.vstack(out_diags), tuple(out_offsets), self.dims)

    def __matmul__(self, other: Array | SparseQArray) -> Array | SparseQArray:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=True, other=other)

        elif isinstance(other, SparseQArray):
            return self._matmul_dia(other=other)

        return NotImplemented

    def __rmatmul__(self, other: Array) -> Array:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=False, other=other)

        return NotImplemented


def _check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'QArrays have incompatible dimensions. Got {dims1} and {dims2}.'
        )
