from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ['SparseDIA', 'to_sparse', 'to_dense']


class SparseDIA(eqx.Module):
    diags: jax.Array
    offsets: tuple[int, ...] = eqx.field(static=True)

    @property
    def ndim(self) -> int:
        return len(self.diags.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.diags.shape

    def dag(self) -> SparseDIA:
        dag_offsets = []
        dag_diags = []

        for diag, offset in zip(self.diags, self.offsets):
            dag_offsets.append(-1 * offset)
            dag_diags.append(jnp.roll(diag, -offset))

        dag_diags = jnp.vstack(dag_diags)

        return SparseDIA(dag_diags, dag_offsets)

    def _cleanup(self, diags: Array, offsets: tuple[int]) -> tuple[Array, tuple[int]]:
        diags = jnp.asarray(diags)
        offsets = jnp.asarray(offsets)
        mask = jnp.any(diags != 0, axis=1)
        diags = jnp.where(mask[:, None], diags, 0)
        offsets = jnp.where(mask, offsets, -1)
        unique_offsets, indices = jnp.unique(offsets, return_inverse=True)
        result = jnp.zeros((len(unique_offsets), diags.shape[1]))

        for i in range(len(unique_offsets)):
            result = result.at[i].set(jnp.sum(diags[indices == i], axis=0))

        return result, tuple([offset.item() for offset in unique_offsets])

    def to_dense(self) -> Array:
        return to_dense(self)

    @functools.partial(jax.jit, static_argnums=(1,))
    def _matmul_dense(self, left_matmul: bool, other: Array) -> Array:
        N = other.shape[0]
        out = jnp.zeros_like(other)
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, offset)
            end = min(N, N + offset)
            top = max(0, -offset)
            bottom = top + end - start
            if left_matmul:
                out = out.at[top:bottom, :].add(
                    diag[start:end, None] * other[start:end, :]
                )
            else:
                out = out.at[:, start:end].add(
                    jnp.transpose(
                        diag[start:end, None] * jnp.transpose(other[:, top:bottom])
                    )
                )
        return out

    def _matmul_dia(self, other: SparseDIA) -> tuple[Array, list[Array]]:
        N = other.diags.shape[1]

        out_diags = []
        out_offsets = []

        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                out_diag = jnp.zeros_like(self_diag)

                sA, sB = max(0, -other_offset), max(0, other_offset)
                eA, eB = min(N, N - other_offset), min(N, N + other_offset)

                out_diag = out_diag.at[sB:eB].add(self_diag[sA:eA] * other_diag[sB:eB])

                new_offset = self_offset + other_offset

                out_diags.append(out_diag)
                out_offsets.append(new_offset)

        out_diags = jnp.vstack(out_diags)

        return out_diags, out_offsets

    @jax.jit
    def _add_dense(self, other: Array) -> Array:
        """Sparse + Dense only using information about diagonals & offsets."""
        N = other.shape[0]
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, offset)
            end = min(N, N + offset)

            s = max(0, abs(offset))
            e = min(N, N + abs(offset))

            i = jnp.arange(e - s) if offset > 0 else jnp.arange(s, e)
            j = jnp.arange(s, e) if offset > 0 else jnp.arange(e - s)

            other = other.at[i, j].add(diag[start:end])

        return other

    def _add_dia(self, other: SparseDIA) -> tuple[Array, list[Array]]:
        self_offsets = list(self.offsets)
        other_offsets = list(other.offsets)

        offset_to_diag = dict(zip(self_offsets, self.diags))

        for other_offset, other_diag in zip(other_offsets, other.diags):
            if other_offset in offset_to_diag:
                offset_to_diag[other_offset] += other_diag
            else:
                offset_to_diag[other_offset] = other_diag

        out_offsets = sorted(offset_to_diag.keys())
        out_diags = jnp.array([offset_to_diag[offset] for offset in out_offsets])

        return out_diags, out_offsets

    @jax.jit
    def _mul_dense(self, other: Array) -> SparseDIA:
        out_diags = []
        out_offsets = []

        for self_offset, self_diag in zip(self.offsets, self.diags):
            other_diag = jnp.diagonal(other, self_offset)

            if self_offset > 0:
                padding = (self_offset, 0)
            elif self_offset < 0:
                padding = (0, -self_offset)
            else:
                padding = (0, 0)

            other_diag = jnp.pad(
                other_diag, padding, mode='constant', constant_values=0
            )
            out_diag = other_diag * self_diag
            out_diags.append(out_diag)
            out_offsets.append(self_offset)

        out_diags = jnp.vstack(out_diags)
        return out_diags, out_offsets

    @jax.jit
    def _mul_dia(self, other: SparseDIA) -> tuple[Array, list[Array]]:
        out_diags = []
        out_offsets = []

        for self_offset, self_diag in zip(self.offsets, self.diags):
            for other_offset, other_diag in zip(other.offsets, other.diags):
                if self_offset != other_offset:
                    continue
                out_diag = jnp.zeros_like(self_diag)
                out_diag = out_diag.at[:].add(self_diag * other_diag)
                out_diags.append(out_diag)
                out_offsets.append(other_offset)

        out_diags = jnp.vstack(out_diags)

        return out_diags, out_offsets

    ### DUNDERS ###

    def __call__(self) -> Array:
        return self

    def __matmul__(self, other: Array | SparseDIA) -> Array | SparseDIA:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=True, other=other)

        elif isinstance(other, SparseDIA):
            diags, offsets = self._matmul_dia(other=other)
            # diags, offsets = self._cleanup(diags, offsets)
            return SparseDIA(diags, tuple(offsets))

        return NotImplemented

    def __rmatmul__(self, other: Array) -> Array:
        if isinstance(other, Array):
            return self._matmul_dense(left_matmul=False, other=other)

        return NotImplemented

    def __add__(self, other: Array | SparseDIA | complex) -> Array | SparseDIA:
        if isinstance(other, Array):
            return self._add_dense(other)

        elif isinstance(other, SparseDIA):
            diags, offsets = self._add_dia(other=other)
            return SparseDIA(diags, tuple(offsets))

        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            else:
                return self.__add__(other)

        return NotImplemented

    def __radd__(self, other: Array) -> Array:
        if isinstance(other, Array):
            return self._add_dense(other)

        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            else:
                return self.__add__(other)

        return NotImplemented

    def __sub__(self, other: Array | SparseDIA) -> Array | SparseDIA:
        if isinstance(other, Array):
            return self._add_dense(-1 * other)

        elif isinstance(other, SparseDIA):
            new_other = SparseDIA(-1 * other.diags, other.offsets)
            diags, offsets = self._add_dia(other=new_other)
            return SparseDIA(diags, tuple(offsets))

        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            else:
                return self.__add__(other)

        return NotImplemented

    def __rsub__(self, other: Array) -> Array:
        if isinstance(other, Array):
            return self._add_dense(-1 * other)

        elif isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            else:
                return self.__add__(other)

        return NotImplemented

    def __mul__(self, other: Array | SparseDIA) -> Array | SparseDIA:
        if isinstance(other, Array):
            diags, offsets = self._mul_dense(other)
            return SparseDIA(diags, tuple([offset.item() for offset in offsets]))

        elif isinstance(other, SparseDIA):
            diags, offsets = self._mul_dia(other)
            return SparseDIA(diags, tuple([offset.item() for offset in offsets]))

        elif isinstance(other, (complex, float)):
            diags, offsets = other * self.diags, self.offsets
            return SparseDIA(diags, offsets)

        return NotImplemented

    def __rmul__(self, other: Array) -> Array:
        if isinstance(other, Array):
            if other.shape == (1, 1):
                diags, offsets = other * self.diags, self.offsets
                return SparseDIA(diags, offsets)
            else:
                diags, offsets = self._mul_dense(other)
                return SparseDIA(diags, tuple([offset.item() for offset in offsets]))
        elif isinstance(other, (complex, float)):
            diags, offsets = other * self.diags, self.offsets
            return SparseDIA(diags, offsets)

        return NotImplemented


def to_dense(sparse: SparseDIA) -> Array:
    r"""Returns the input matrix in the Dense format.

    Parameters:
        sparse: A sparse matrix, containing diagonals and their offsets.

    Returns:
        Array: A dense matrix representation of the input sparse matrix.
    """
    N = sparse.diags.shape[1]
    out = jnp.zeros((N, N))
    for offset, diag in zip(sparse.offsets, sparse.diags):
        start = max(0, offset)
        end = min(N, N + offset)
        out += jnp.diag(diag[start:end], k=offset)
    return out


def to_sparse(other: Array) -> SparseDIA:
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

    n = other.shape[0]
    offset_range = 2 * n - 1
    offset_center = offset_range // 2

    # maybe the best way is to check all conditions but for now stick to 'not Array'
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

    return SparseDIA(diagonals, offsets)
