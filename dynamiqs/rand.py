from jax import numpy as jnp, Array
import jax

from dynamiqs.utils import dag  # todo: clean this dependency


def hermitian(n: int, key=None) -> Array:
    key = key if key is not None else jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    mat = 2 * jax.random.uniform(key1, (n, n)) - 1
    mat = mat + 1j * (2 * jax.random.uniform(key2, (n, n)) - 1)
    mat = 0.5 * (mat + dag(mat))
    return mat


def dm(n: int, key=None) -> Array:
    mat = hermitian(n, key)
    mat = mat.at[jnp.diag_indices(n)].set(jnp.abs(jnp.diag(mat)) + jnp.sqrt(2) * n)
    mat /= jnp.trace(mat)
    return mat


def ket(n: int, key=None) -> Array:
    key = key if key is not None else jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    vec = jax.random.normal(key1, (n, 1)) + 1j * jax.random.normal(key2, (n, 1))
    vec /= jnp.linalg.norm(vec)
    return vec
