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
from qutip import Qobj

from .dense_qarray import DenseQArray
from .qarray import QArray

__all__ = ['SparseDIAQArray', 'to_dense', 'to_sparse_dia', 'stack']


class SparseDIAQArray(QArray):
    dims: tuple[int, ...]
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array

    def __init__(self, dims: tuple[int, ...], offsets: tuple[int, ...], diags: Array):
        self.dims = dims
        self.offsets = offsets
        if isinstance(diags, list):
            self.diags = jnp.stack(diags)
        elif isinstance(diags, Array) and len(diags.shape) == 2:
            self.diags = jnp.expand_dims(diags, axis=0)
        else:
            self.diags = diags

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        depth = self.diags.shape[0]
        N = self.diags.shape[-1]
        return (depth, N, N)

    @property
    def I(self) -> QArray:  # noqa: E743
        diags = jnp.ones((1, self.shape[-1]))
        offsets = (0,)
        return SparseDIAQArray(self.dims, offsets, diags)

    def conj(self) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags.conj())

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
        return SparseDIAQArray(self.dims, offsets, diags)

    def _trace(self) -> Array:
        main_diag_mask = jnp.asarray(self.offsets) == 0
        return jax.lax.cond(
            jnp.any(main_diag_mask),
            lambda: jnp.sum(self.diags[jnp.argmax(main_diag_mask)]).astype(jnp.float32),
            lambda: jnp.array(0.0, dtype=jnp.float32),
        )

    def norm(self) -> Array:
        return self._trace().real

    def reshape(self, *shape: int) -> QArray:  # noqa: ARG002
        return NotImplemented

    def broadcast_to(self, *shape: int) -> QArray:  # noqa: ARG002
        return NotImplemented

    def ptrace(self, keep: tuple[int, ...]) -> QArray:  # noqa: ARG002
        return NotImplemented

    def isket(self) -> bool:
        return NotImplemented

    def isbra(self) -> bool:
        return NotImplemented

    def isdm(self) -> bool:
        return NotImplemented

    def isherm(self) -> bool:
        return NotImplemented

    def toket(self) -> QArray:
        return NotImplemented

    def tobra(self) -> QArray:
        return NotImplemented

    def todm(self) -> QArray:
        return NotImplemented

    def to_numpy(self) -> np.ndarray:
        return NotImplemented

    def to_qutip(self) -> Qobj:
        return NotImplemented

    def to_jax(self) -> Array:
        return NotImplemented

    def __mul__(self, other: Array | SparseDIAQArray) -> QArray:
        if isinstance(other, (complex, Scalar)):
            diags, offsets = other * self.diags, self.offsets
            return SparseDIAQArray(offsets, diags, self.dims)
        elif isinstance(other, Array):
            return self._mul_dense(other)
        elif isinstance(other, SparseDIAQArray):
            _check_compatible_dims(self.dims, other.dims)
            return self._mul_sparse(other)

        return NotImplemented

    def _mul_dense(self, other: Array) -> QArray:
        diags = jnp.transpose(self.diags, (1, 0, 2))
        depth = diags.shape[1]
        N = other.shape[-1]
        out_diags = jnp.zeros_like(self.diags)
        for i, (offset, batched_diags) in enumerate(zip(self.offsets, diags)):
            start = max(0, offset)
            end = min(N, N + offset)
            other_diag = jnp.diagonal(other, offset)
            other_diag_tensor = jnp.broadcast_to(
                other_diag, (depth, other_diag.shape[0])
            )
            out_diags = out_diags.at[:, i, start:end].set(
                other_diag_tensor * batched_diags[:, start:end]
            )

        return SparseDIAQArray(self.dims, self.offsets, out_diags)

    def _mul_sparse(self, other: SparseDIAQArray) -> QArray:
        diags = jnp.transpose(self.diags, (1, 0, 2))
        other_diags = jnp.transpose(other.diags, (1, 0, 2))
        N, M = other.diags.shape[-1], diags.shape[1]
        out_offsets, out_diags = [], []
        for self_offset, batched_diags in zip(self.offsets, diags):
            for other_offset, other_diag in zip(other.offsets, other_diags):
                if self_offset != other_offset:
                    continue
                out_offsets.append(other_offset)
                out_diags.append(batched_diags * jnp.broadcast_to(other_diag, (M, N)))

        return SparseDIAQArray(
            self.dims,
            tuple(out_offsets),
            jnp.transpose(jnp.stack(out_diags), (1, 0, 2)),
        )

    def __add__(self, other: ScalarLike | ArrayLike) -> QArray:
        if isinstance(other, ScalarLike):
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
            _check_compatible_dims(self.dims, other.dims)
            return self._add_sparse(other)

        return NotImplemented

    def _add_sparse(self, other: SparseDIAQArray) -> QArray:
        diags = jnp.transpose(self.diags, (1, 0, 2))
        other_diags = jnp.transpose(other.diags, (1, 0, 2))
        N, M = diags.shape[-1], diags.shape[1]
        out_offsets_diags = dict(zip(self.offsets, diags))
        for other_offset, other_diag in zip(other.offsets, other_diags):
            if other_offset in out_offsets_diags:
                out_offsets_diags[other_offset] += jnp.broadcast_to(other_diag, (M, N))
            else:
                out_offsets_diags[other_offset] = jnp.broadcast_to(other_diag, (M, N))
        out_offsets = tuple(sorted(out_offsets_diags.keys()))
        out_diags = jnp.transpose(
            jnp.stack([out_offsets_diags[offset] for offset in out_offsets]), (1, 0, 2)
        )

        return SparseDIAQArray(self.dims, out_offsets, out_diags)

    def __matmul__(self, other: Array | SparseDIAQArray) -> QArray:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=True, other=other)
        elif isinstance(other, SparseDIAQArray):
            return self._matmul_dia(other=other)

        return NotImplemented

    def __rmatmul__(self, other: Array) -> QArray:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=False, other=other)

        return NotImplemented

    def _matmul_dense(self, left_matmul: bool, other: Array) -> QArray:
        diags = jnp.transpose(self.diags, (1, 2, 0))
        depth = diags.shape[-1]
        N = other.shape[-1]
        tensor_other = jnp.broadcast_to(other, (depth, N, N))
        out_tensor = jnp.zeros_like(tensor_other)
        for offset, batched_diags in zip(self.offsets, diags):
            start = max(0, offset)
            end = min(N, N + offset)
            top = max(0, -offset)
            bottom = top + end - start

            def left_case(
                out_tensor: Array,
                top: int = top,
                bottom: int = bottom,
                start: int = start,
                end: int = end,
                b_diags: Array = batched_diags,
            ) -> Array:
                return out_tensor.at[:, top:bottom, :].add(
                    b_diags.T[:, start:end, jnp.newaxis] * tensor_other[:, start:end, :]
                )

            def right_case(
                out_tensor: Array,
                top: int = top,
                bottom: int = bottom,
                start: int = start,
                end: int = end,
                b_diags: Array = batched_diags,
            ) -> Array:
                return out_tensor.at[:, :, start:end].add(
                    jnp.transpose(b_diags.T[:, start:end, None], (0, 2, 1))
                    * tensor_other[:, :, top:bottom]
                )

            out_tensor = jax.lax.cond(left_matmul, left_case, right_case, out_tensor)
        return DenseQArray(self.dims, out_tensor)

    def _matmul_dia(self, other: SparseDIAQArray) -> QArray:
        diags = jnp.transpose(self.diags, (1, 0, 2))
        other_diags = jnp.transpose(other.diags, (1, 0, 2))
        N = other.diags.shape[-1]
        M = diags.shape[1]

        diag_dict = defaultdict(lambda: jnp.zeros((M, N)))
        for self_offset, batched_diags in zip(self.offsets, diags):
            for other_offset, other_diag in zip(other.offsets, other_diags):
                result_offset = self_offset + other_offset
                if abs(result_offset) > N - 1:
                    continue
                sA, sB = max(0, -other_offset), max(0, other_offset)
                eA, eB = min(N, N - other_offset), min(N, N + other_offset)
                other_diag_broadcasted = jnp.broadcast_to(
                    other_diag, (batched_diags.shape[0], other_diag.shape[-1])
                )
                diag_dict[result_offset] = (
                    diag_dict[result_offset]
                    .at[:, sB:eB]
                    .add(batched_diags[:, sA:eA] * other_diag_broadcasted[:, sB:eB])
                )

        out_offsets = sorted(diag_dict.keys())
        out_diags = [diag_dict[offset] for offset in out_offsets]

        if len(out_offsets) == 0:
            return SparseDIAQArray(self.dims, (0,), jnp.zeros((self.shape[0], N)))

        return SparseDIAQArray(
            self.dims,
            tuple(out_offsets),
            jnp.transpose(jnp.stack(out_diags), (1, 0, 2)),
        )

    def __and__(self, y: QArray) -> QArray:
        return NotImplemented

    def __pow__(self, y: Scalar) -> QArray:
        return NotImplemented


def _check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'QArrays have incompatible dimensions. Got {dims1} and {dims2}.'
        )


def to_dense(x: SparseDIAQArray) -> DenseQArray:
    r"""Convert a sparse `QArray` into a dense `Qarray`.

    Args:
        x: A sparse matrix, containing diagonals and their offsets.

    Returns:
        Array: A dense matrix representation of the input sparse matrix.
    """
    depth = x.diags.shape[0]
    N = x.diags.shape[-1]
    out = jnp.zeros((depth, N, N))
    diags = jnp.transpose(x.diags, (1, 0, 2))
    for i, (offset, diag) in enumerate(zip(x.offsets, diags)):
        start = max(0, offset)
        end = min(N, N + offset)
        tensor = jnp.stack(
            [jnp.diag(diag[i, start:end], k=offset) for i in range(3)], axis=0
        )
        out = out.at[...].add(tensor)
    return DenseQArray(x.dims, out)


def _find_offsets(x: ArrayLike) -> tuple[int, ...]:
    indices = np.nonzero(x)
    return tuple(np.unique(indices[2] - indices[1]))


@functools.partial(jax.jit, static_argnums=(0,))
def _construct_diags(offsets: tuple[int, ...], x: ArrayLike) -> Array:
    depth = x.shape[0]
    N = x.shape[-1]
    diags = jnp.zeros((depth, len(offsets), N))
    for i, offset in enumerate(offsets):
        start = max(0, offset)
        end = min(N, N + offset)
        diagonals = jnp.diagonal(x, offset=offset, axis1=1, axis2=2)
        diags = diags.at[:, i, start:end].set(diagonals)

    return diags


def to_sparse_dia(x: DenseQArray | Array) -> SparseDIAQArray:
    r"""Returns the input matrix in the `SparseDIAQArray` format.

    Args:
        x: Matrix to turn from dense to SparseDIA format.

    Returns:
        `SparseDIAQArray` object
    """
    concrete_or_error(None, x, '`to_sparse_dia` does not support tracing.')
    batched_x = jnp.stack(x)
    offsets = _find_offsets(batched_x)
    diags = _construct_diags(offsets, batched_x)
    return SparseDIAQArray(x.dims, offsets, diags)


def stack(x_list: list) -> SparseDIAQArray:
    diags = jnp.stack([x.diags for x in x_list], axis=0)
    offsets = x_list[0].offsets
    dims = x_list[0].dims
    return SparseDIAQArray(dims, offsets, diags)
