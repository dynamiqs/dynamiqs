from __future__ import annotations

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from dynamiqs.utils import dag  # todo: clean this dependency

from .utils.array_types import dtype_complex_to_real, get_cdtype


def matrix(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    rdtype = dtype_complex_to_real(get_cdtype(dtype))

    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape, dtype=rdtype)
    y = jax.random.normal(key2, shape, dtype=rdtype)
    return x + 1j * y


def herm(
    key: PRNGKeyArray,
    dim: int,
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # hermitian
    x = matrix(key, (dim, dim), dtype=dtype)
    return 0.5 * (x + dag(x))


def psd(
    key: PRNGKeyArray,
    dim: int,
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # positive semi-definite
    x = matrix(key, (dim, dim), dtype=dtype)
    return x @ dag(x)


def dm(
    key: PRNGKeyArray,
    dim: int,
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    x = psd(key, dim, dtype=dtype)
    x /= x.trace().real
    return x


def ket(
    key: PRNGKeyArray,
    dim: int,
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    x = matrix(key, (dim, 1), dtype=dtype)
    x /= jnp.linalg.norm(x).real
    return x
