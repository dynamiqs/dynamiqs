from __future__ import annotations

import functools
import warnings
from collections import defaultdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import concrete_or_error
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from .dense_qarray import DenseQArray
from .qarray import QArray

__all__ = ['SparseQArray', 'to_dense', 'to_sparse']


class SparseQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        N = self.diags.shape[-1]
        return (N, N)

    @property
    def I(self) -> QArray:  # noqa: E743
        diags = jnp.ones((1, self.shape[-1]))
        offsets = (0,)
        return SparseQArray(diags, offsets, self.dims)

    def conj(self) -> QArray:
        return SparseQArray(self.diags.conj(), self.offsets, self.dims)

    def dag(self) -> QArray:
        N = self.shape[-1]
        diags = jnp.zeros_like(self.diags)
        for i, (self_offset, self_diag) in enumerate(zip(self.offsets, self.diags)):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            diags = diags.at[i, start - self_offset : end - self_offset].set(
                self_diag[start:end].conj()
            )
        offsets = tuple(-x for x in self.offsets)
        return SparseQArray(diags, offsets, self.dims)

    def _trace(self) -> Array:
        main_diag_mask = jnp.asarray(self.offsets) == 0
        return jax.lax.cond(
            jnp.any(main_diag_mask),
            lambda: jnp.sum(self.diags[jnp.argmax(main_diag_mask)]).astype(jnp.float32),
            lambda: jnp.array(0.0, dtype=jnp.float32),
        )

    def norm(self) -> Array:
        return self._trace().real

    def unit(self) -> QArray:
        return SparseQArray(self.diags / self.norm(), self.offsets, self.dims)

    def __add__(self, other: ScalarLike | ArrayLike) -> QArray:
        if isinstance(other, ScalarLike):
            if other == 0:
                return self
            warnings.warn(
                '`to_dense` was called on a SparseQArray due to addition with a scalar.'
                ' The array is no longer in sparse format.',
                stacklevel=2,
            )
            return self.to_dense() + other
        elif isinstance(other, ArrayLike):
            warnings.warn(
                '`to_dense` was called on a SparseQArray due to addition with a dense'
                ' array. The array is no longer in sparse format.',
                stacklevel=2,
            )
            return self.to_dense() + other
        elif isinstance(other, SparseQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._add_sparse(other)

        return NotImplemented

    def _add_sparse(self, other: SparseQArray) -> QArray:
        out_offsets_diags = dict(zip(self.offsets, self.diags))
        for other_offset, other_diag in zip(other.offsets, other.diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] += other_diag
            else:
                out_offsets_diags[other_offset] = other_diag

        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.stack([out_offsets_diags[offset] for offset in out_offsets])

        return SparseQArray(out_offsets, out_diags, self.dims)

    def __mul__(self, other: Array | SparseQArray) -> QArray:
        if isinstance(other, (complex, Scalar)):
            diags, offsets = other * self.diags, self.offsets
            return SparseQArray(offsets, diags, self.dims)
        elif isinstance(other, Array):
            return self._mul_dense(other)
        elif isinstance(other, SparseQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._mul_sparse(other)

        return NotImplemented

    def _mul_dense(self, other: Array) -> QArray:
        N = other.shape[0]
        out_diags = jnp.zeros_like(self.diags)
        for i, (self_offset, self_diag) in enumerate(zip(self.offsets, self.diags)):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            other_diag = jnp.diagonal(other, self_offset)
            out_diags = out_diags.at[i, start:end].set(
                other_diag * self_diag[start:end]
            )

        return SparseQArray(self.offsets, out_diags, self.dims)

    def _mul_sparse(self, other: SparseQArray) -> QArray:
        out_diags, out_offsets = [], []
        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                if self_offset != other_offset:
                    continue
                out_diags.append(self_diag * other_diag)
                out_offsets.append(other_offset)

        return SparseQArray(tuple(out_offsets), jnp.stack(out_diags), self.dims)

    def _matmul_dense(self, left_matmul: bool, other: Array) -> QArray:
        N = other.shape[0]
        out = jnp.zeros_like(other)
        for self_offset, self_diag in zip(self.offsets, self.diags):
            start = max(0, self_offset)
            end = min(N, N + self_offset)
            top = max(0, -self_offset)
            bottom = top + end - start

            def left_case(
                out: Array,
                top: int = top,
                bottom: int = bottom,
                start: int = start,
                end: int = end,
                diag: Array = self_diag,
            ) -> Array:
                return out.at[top:bottom, :].add(
                    diag[start:end, None] * other[start:end, :]
                )

            def right_case(
                out: Array,
                top: int = top,
                bottom: int = bottom,
                start: int = start,
                end: int = end,
                diag: Array = self_diag,
            ) -> Array:
                return out.at[:, start:end].add(
                    diag[start:end, None].T * other[:, top:bottom]
                )

            out = jax.lax.cond(left_matmul, left_case, right_case, out)
        return out

    def _matmul_dia(self, other: SparseQArray) -> QArray:
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

        return SparseQArray(jnp.vstack(out_diags), tuple(out_offsets), self.dims)

    def __matmul__(self, other: Array | SparseQArray) -> QArray:
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


def to_dense(x: SparseQArray) -> DenseQArray:
    r"""Convert a sparse `QArray` into a dense `Qarray`.

    Args:
        x: A sparse matrix, containing diagonals and their offsets.

    Returns:
        Array: A dense matrix representation of the input sparse matrix.
    """
    N = x.shape[-1]
    out = jnp.zeros((N, N))
    for offset, diag in zip(x.offsets, x.diags):
        start = max(0, offset)
        end = min(N, N + offset)
        out += jnp.diag(diag[start:end], k=offset)
    return out


def find_offsets(other: ArrayLike) -> tuple[int, ...]:
    indices = np.nonzero(other)
    return tuple(np.unique(indices[1] - indices[0]))


@functools.partial(jax.jit, static_argnums=(0,))
def produce_dia(offsets: tuple[int, ...], other: ArrayLike) -> Array:
    n = other.shape[0]
    diags = jnp.zeros((len(offsets), n))

    for i, offset in enumerate(offsets):
        start = max(0, offset)
        end = min(n, n + offset)
        diagonal = jnp.diagonal(other, offset=offset)
        diags = diags.at[i, start:end].set(diagonal)

    return diags


def to_sparse(x: DenseQArray | Array) -> SparseQArray:
    r"""Returns the input matrix in the `SparseQArray` format.

    Args:
        x: Matrix to turn from dense to SparseDIA format.

    Returns:
        `SparseQArray` object
    """
    concrete_or_error(None, x, '`to_sparse` does not support tracing.')
    offsets = find_offsets(x)
    diags = produce_dia(offsets, x)
    return SparseQArray(diags, offsets, x.dims)
