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

    @functools.partial(jax.jit, static_argnums=(0,))
    def matmul(
        self,
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
            out = out.at[top:bottom, :].add(
                diag[start:end, None] * matrix[start:end, :]
            )
        return out

    def dag(self):
        """Returns the hermitian conjugate, call to_dense() to visualize."""
        self.offsets = tuple(-1 * jnp.array(self.offsets))
        self.diags = jnp.conjugate(self.diags)

    @functools.partial(jax.jit, static_argnums=(0,))
    def matadd(self, matrix: Array) -> Array:
        """Sparse + Dense only using information about diagonals & offsets."""
        N = matrix.shape[0]
        for offset, diag in zip(self.offsets, self.diags):
            start = max(0, abs(offset))
            end = min(N, N + abs(offset))
            i = jnp.arange(end - start) if offset > 0 else jnp.arange(start, end)
            j = jnp.arange(start, end) if offset > 0 else jnp.arange(end - start)

            matrix = matrix.at[i, j].add(diag[start:end])

        return matrix
