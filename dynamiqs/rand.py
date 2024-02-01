import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from dynamiqs.utils import dag  # todo: clean this dependency


def matrix(dims: tuple[int, ...], key) -> Array:
    key1, key2 = jax.random.split(key, 2)
    mat = jax.random.normal(key1, dims) + 1j * jax.random.normal(key2, dims)
    return mat


def hermitian(dims: tuple[int, ...], key) -> Array:
    mat = matrix((np.prod(dims), np.prod(dims)), key=key)
    return 0.5 * (mat + dag(mat))


def dm(dims: tuple[int, ...], key) -> Array:
    mat = hermitian(dims, key)
    mat = mat.at[jnp.diag_indices(np.prod(dims))].set(
        jnp.abs(jnp.diag(mat)) + jnp.sqrt(2) * np.prod(dims)
    )
    mat /= jnp.trace(mat)
    return mat


def ket(dims: tuple[int, ...], key) -> Array:
    vec = matrix((np.prod(dims), 1), key)
    vec /= jnp.linalg.norm(vec)
    return vec
