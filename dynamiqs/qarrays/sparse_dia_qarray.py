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
from jaxtyping import Array, ArrayLike, ScalarLike
from qutip import Qobj

from .dense_qarray import DenseQArray
from .types import QArray, QArrayLike, asqarray

__all__ = ['SparseDIAQArray', 'to_dense', 'to_sparse_dia']


class SparseDIAQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array = eqx.field(converter=jnp.asarray)

    def __post_init__(self):
        # check that diagonals contain zeros outside the bounds of the matrix
        for offset, diag in zip(self.offsets, self.diags):
            if (offset < 0 and jnp.any(diag[offset:] != 0)) or (
                offset > 0 and jnp.any(diag[:offset] != 0)
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
        return (N, N)

    @property
    def mT(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

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

    def reshape(self, *shape: int) -> QArray:
        raise NotImplementedError

    def broadcast_to(self, *shape: int) -> QArray:
        raise NotImplementedError

    def ptrace(self, keep: tuple[int, ...]) -> QArray:  # noqa: ARG002
        return NotImplemented

    def powm(self):
        raise NotImplementedError

    def expm(self):
        raise NotImplementedError

    def unit(self) -> SparseDIAQArray:
        return SparseDIAQArray(self.offsets, self.diags / self.norm(), self.dims)

    def norm(self) -> Array:
        raise NotImplementedError

    def trace(self) -> Array:
        main_diag_mask = jnp.asarray(self.offsets) == 0
        return jax.lax.cond(
            jnp.any(main_diag_mask),
            lambda: jnp.sum(self.diags[jnp.argmax(main_diag_mask)]).astype(jnp.float32),
            lambda: jnp.array(0.0, dtype=jnp.float32),
        )

    def sum(self):
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

    def __len__(self):
        raise NotImplementedError

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        raise NotImplementedError

    def __mul__(self, other: Array | SparseDIAQArray) -> QArray:
        if isinstance(other, get_args(ScalarLike)):
            diags, offsets = other * self.diags, self.offsets
            return SparseDIAQArray(offsets=offsets, diags=diags, dims=self.dims)
        elif isinstance(other, Array):
            return self._mul_dense(other)
        elif isinstance(other, SparseDIAQArray):
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

        return SparseDIAQArray(self.dims, self.offsets, out_diags)

    def _mul_sparse(self, other: SparseDIAQArray) -> QArray:
        out_diags, out_offsets = [], []
        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                if self_offset != other_offset:
                    continue
                out_diags.append(self_diag * other_diag)
                out_offsets.append(other_offset)

        return SparseDIAQArray(self.dims, tuple(out_offsets), jnp.stack(out_diags))

    def __truediv__(self, y: QArrayLike) -> QArray:
        raise NotImplementedError

    def __add__(self, other: ScalarLike | ArrayLike | QArray) -> QArray:
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
        elif isinstance(other, DenseQArray):
            _check_compatible_dims(self.dims, other.dims)
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

            # todo: replace by a dump "if"
            out = jax.lax.cond(left_matmul, left_case, right_case, out)

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
        N = other.diags.shape[-1]
        out_offsets = jnp.ravel(
            jnp.asarray(self.offsets) * N + jnp.asarray(other.offsets)[:, None],
            order='F',
        )
        out_diags = jnp.kron(self.diags, other.diags)

        offsets = tuple(out_offsets)
        diags = out_diags
        offsets = jnp.sort(jnp.asarray(offsets))

        unique_offsets, inverse_indices = np.unique(offsets, return_inverse=True)
        count = unique_offsets.shape[0]
        out_diags = jnp.zeros((count, diags.shape[-1]))
        for i in range(count):
            mask = inverse_indices == i
            out_diags = out_diags.at[i, :].set(jnp.sum(diags[mask, :], axis=0))
        return SparseDIAQArray(
            offsets=tuple(o.item() for o in unique_offsets),
            diags=out_diags,
            dims=self.dims + other.dims,
        )

    def __and__(self, other: QArrayLike) -> QArray:
        if isinstance(other, SparseDIAQArray):
            return self._kronecker_dia(other=other)

        elif isinstance(other, get_args(QArrayLike)):
            return self.to_dense() & asqarray(other)

        return NotImplemented

    def _pow(self, power: int) -> QArray:  # noqa: ARG002
        return NotImplemented

    def __getitem__(self, key: int | slice) -> QArray:
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
    N = x.shape[-1]
    out = jnp.zeros((N, N))
    for offset, diag in zip(x.offsets, x.diags):
        start = max(0, offset)
        end = min(N, N + offset)
        out += jnp.diag(diag[start:end], k=offset)
    return DenseQArray(x.dims, out)


def _find_offsets(other: ArrayLike) -> tuple[int, ...]:
    indices = np.nonzero(other)
    return tuple(np.unique(indices[1] - indices[0]))


@functools.partial(jax.jit, static_argnums=(0,))
def _construct_diags(offsets: tuple[int, ...], other: ArrayLike) -> Array:
    n = other.shape[0]
    diags = jnp.zeros((len(offsets), n))

    for i, offset in enumerate(offsets):
        start = max(0, offset)
        end = min(n, n + offset)
        diagonal = jnp.diagonal(other, offset=offset)
        diags = diags.at[i, start:end].set(diagonal)

    return diags


def to_sparse_dia(x: DenseQArray | ArrayLike | SparseDIAQArray) -> SparseDIAQArray:
    r"""Returns the input matrix in the `SparseDIAQArray` format.

    Args:
        x: Matrix to turn from dense to SparseDIA format.

    Returns:
        `SparseDIAQArray` object
    """
    if isinstance(x, SparseDIAQArray):
        return x
    elif isinstance(x, DenseQArray):
        dims = x.dims
        x = x.to_jax()
    elif isinstance(x, get_args(ArrayLike)):
        dims = (x.shape[-1],)
    else:
        raise TypeError

    concrete_or_error(None, x, '`to_sparse_dia` does not support tracing.')
    offsets = _find_offsets(x)
    diags = _construct_diags(offsets, x)
    return SparseDIAQArray(dims=dims, offsets=offsets, diags=diags)
