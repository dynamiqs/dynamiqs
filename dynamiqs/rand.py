from __future__ import annotations

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from dynamiqs.utils import dag  # todo: clean this dependency

from .utils.array_types import dtype_complex_to_real, get_cdtype


def matrix(
    shape: tuple[int, ...],
    *,
    key: PRNGKeyArray | None = None,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # seed PRNG key if None
    if key is None:
        key = jax.random.PRNGKey(42)

    rdtype = dtype_complex_to_real(get_cdtype(dtype))

    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape, dtype=rdtype)
    y = jax.random.normal(key2, shape, dtype=rdtype)
    return x + 1j * y


def herm(
    dim: int,
    *,
    key: PRNGKeyArray | None = None,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # hermitian
    x = matrix((dim, dim), key=key, dtype=dtype)
    return 0.5 * (x + dag(x))


def psd(
    dim: int,
    *,
    key: PRNGKeyArray | None = None,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # positive semi-definite
    x = matrix((dim, dim), key=key, dtype=dtype)
    return x @ dag(x)


def dm(
    dim: int,
    *,
    key: PRNGKeyArray | None = None,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    x = psd(dim, key=key, dtype=dtype)
    x /= x.trace().real
    return x


def ket(
    dim: int,
    *,
    key: PRNGKeyArray | None = None,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    x = matrix((dim, 1), key=key, dtype=dtype)
    x /= jnp.linalg.norm(x).real
    return x
