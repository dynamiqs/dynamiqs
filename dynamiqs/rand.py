from jax import numpy as jnp, Array
import jax

from dynamiqs.utils import dag  # todo: clean this dependency
from utils.utils import _prod


def matrix(dims: tuple[int, ...], key) -> Array:
    key1, key2 = jax.random.split(key, 2)
    mat = jax.random.normal(key1, dims) + 1j * jax.random.normal(key2, dims)
    return mat


def hermitian(dims: tuple[int, ...], key) -> Array:
    mat = matrix((_prod(dims), _prod(dims)), key=key)
    return 0.5 * (mat + dag(mat))


def dm(dims: tuple[int, ...], key) -> Array:
    mat = hermitian(dims, key)
    mat = mat.at[jnp.diag_indices(_prod(dims))].set(
        jnp.abs(jnp.diag(mat)) + jnp.sqrt(2) * _prod(dims)
    )
    mat /= jnp.trace(mat)
    return mat


def ket(dims: tuple[int, ...], key) -> Array:
    vec = matrix((_prod(dims), 1), key)
    vec /= jnp.linalg.norm(vec)
    return vec
