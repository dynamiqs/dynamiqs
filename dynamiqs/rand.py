from __future__ import annotations

import jax
from jax import Array
from jaxtyping import PRNGKeyArray

from .utils.utils import dag, unit


def complex(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape)
    y = jax.random.normal(key2, shape)
    return x + 1j * y


def herm(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    # hermitian
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = complex(key, shape)
    return 0.5 * (x + dag(x))


def psd(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    # positive semi-definite
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = complex(key, shape)
    return x @ dag(x)


def dm(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    assert len(shape) >= 2 and shape[-1] == shape[-2]
    x = psd(key, shape)
    return unit(x)


def ket(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    assert len(shape) >= 2 and shape[-1] == 1
    x = complex(key, shape)
    return unit(x)
