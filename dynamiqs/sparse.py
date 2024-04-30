import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array


class SparseDIA:
    def __init__(self, diags: Array, offsets: tuple[int]):
        self.offsets = offsets
        self.diags = diags


    def to_dense(self) -> Array:
        """Turn any set of diagonals & offsets in a full NxN matrix."""
        N = self.diags.shape[1]
        out = jnp.zeros((N, N))
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, offset)
            end = min(N, N + offset)
            out += jnp.diag(diag[start:end], k=offset)
        return out

    @functools.partial(jax.jit, static_argnums=(0,1))
    def _matmul_dense(
        self,
        direction: str,
        matrix: Array,  # If matrix is a vector need to input as column vector.
    ) -> Array:
        """Sparse @ Dense only using information about diagonals & offsets."""
        N = matrix.shape[0]
        out = jnp.zeros_like(matrix)
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, offset)
            end = min(N, N + offset)
            top = max(0, -offset)
            bottom = top + end - start
            if direction=='left':
                out = out.at[top:bottom, :].add(
                    diag[start:end, None] * matrix[start:end, :]
                )
            else:
                out = out.at[:, start:end].add(
                    jnp.transpose(diag[start:end, None] * jnp.transpose(matrix[:, top:bottom]))
                )
        return out

    @functools.partial(jax.jit, static_argnums=(0,1))
    def _matmul_dia(self, matrix) -> Array:
        N = matrix.diags.shape[1]

        vector_list = []
        offset_list = []

        for offset, diag in zip(self.offsets, self.diags):
            for matrix_offset, matrix_diag in zip(matrix.offsets, matrix.diags):

                out_diag = jnp.zeros_like(diag)

                sA, sB = max(0, -matrix_offset), max(0, matrix_offset)
                eA, eB = min(N, N-matrix_offset), min(N, N+matrix_offset)

                out_diag = out_diag.at[sB:eB].add(
                    diag[sA:eA] * matrix_diag[sB:eB]
                )

                new_offset = offset + matrix_offset

                vector_list.append(out_diag)
                offset_list.append(new_offset)

        out = jnp.vstack(vector_list)

        return out, offset_list


    @functools.partial(jax.jit, static_argnums=(0,1))
    def _add_dia(self, matrix):
        self_offsets = list(self.offsets)
        matrix_offsets = list(matrix.offsets)

        offset_to_diag = {offset: diag for offset, diag in zip(self_offsets, self.diags)}

        for offset, diag in zip(matrix_offsets, matrix.diags):
            if offset in offset_to_diag:
                offset_to_diag[offset] += diag
            else:
                offset_to_diag[offset] = diag

        new_offsets = sorted(offset_to_diag.keys())
        new_diagonals = jnp.array([offset_to_diag[offset] for offset in new_offsets])

        return new_diagonals, new_offsets


    def _dag(self):
        """Returns the hermitian conjugate, call to_dense() to visualize."""
        self.offsets = tuple(-1 * jnp.array(self.offsets))
        self.diags = jnp.conjugate(self.diags)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _add_dense(self, matrix: Array) -> Array:
        """Sparse + Dense only using information about diagonals & offsets."""
        N = matrix.shape[0]
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, offset)
            end = min(N, N + offset)

            s = max(0, abs(offset))
            e = min(N, N+abs(offset))

            i = jnp.arange(e - s) if offset > 0 else jnp.arange(s, e)
            j = jnp.arange(s, e) if offset > 0 else jnp.arange(e - s)

            matrix = matrix.at[i, j].add(diag[start:end])

        return matrix
    

    def _cleanup(self, diags, offsets):
        diags = jnp.array(diags)
        offsets = jnp.array(offsets)
        mask = jnp.any(diags != 0, axis=1)
        diags = diags[mask]
        offsets = offsets[mask]
        unique_offsets, indices = jnp.unique(offsets, return_inverse=True)
        result = jnp.zeros((len(unique_offsets), diags.shape[1]))

        for i, offset in enumerate(unique_offsets):
            result = result.at[i].set(jnp.sum(diags[indices == i], axis=0))

        return result, tuple([offset.item() for offset in unique_offsets])


    def __matmul__(self, matrix):
        if isinstance(matrix, Array):
            return self._matmul_dense(direction='left', matrix=matrix)

        elif isinstance(matrix, SparseDIA):
            diags, offsets = self._matmul_dia(matrix=matrix)
            diags, offsets = self._cleanup(diags, offsets)
            return SparseDIA(diags, offsets)
        
        return NotImplemented

    def __rmatmul__(self, matrix):
        if isinstance(matrix, Array):
            return self._matmul_dense(direction='right', matrix=matrix)
        
        return NotImplemented

    def __getitem__(self, index):
        dense = self.to_dense()
        return dense[index]

    def __add__(self, matrix):
        if isinstance(matrix, Array):
            out = self._add_dense(matrix)
            return out

        elif isinstance(matrix, SparseDIA):
            diags, offsets = self._add_dia(matrix=matrix)
            return SparseDIA(diags, tuple([offset.item() for offset in offsets]))

        return NotImplemented

    def __radd__(self, matrix):
        if isinstance(matrix, Array):
            return self._add_dense(matrix)

        return NotImplemented

""" UTILITY FUNCTIONS """

def to_sparse(matrix):
    """Turn a NxN sparse matrix into the amazing SparseDIA format with zero padding"""

    diagonals = []
    offsets = []
    
    n = matrix.shape[0]
    offset_range = 2 * n - 1
    offset_center = offset_range // 2

    for offset in range(-offset_center, offset_center + 1):
        diagonal = jnp.diagonal(matrix, offset=offset)
        if jnp.any(diagonal != 0):
            if offset > 0:
                padding = (offset, 0)
            elif offset < 0:
                padding = (0, -offset)
            else:
                padding = (0, 0)

            padded_diagonal = jnp.pad(diagonal, padding, mode='constant', constant_values=0)
            diagonals.append(padded_diagonal)
            offsets.append(offset)

    diagonals = jnp.array(diagonals)
    offsets = tuple(offsets)
    
    return SparseDIA(diagonals, offsets)
