from __future__ import annotations

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray

from .utils.array_types import dtype_complex_to_real, get_cdtype
from .utils.utils import dag, unit


def complex(
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
    shape: tuple[int, ...],
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # hermitian
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = complex(key, shape, dtype=dtype)
    return 0.5 * (x + dag(x))


def psd(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    # positive semi-definite
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = complex(key, shape, dtype=dtype)
    return x @ dag(x)


def dm(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = psd(key, shape, dtype=dtype)
    return unit(x)


def ket(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: jnp.complex64 | jnp.complex128 | None = None,
) -> Array:
    assert len(shape) >= 2 and shape[-1] == 1
    x = complex(key, shape, dtype=dtype)
    return unit(x)
