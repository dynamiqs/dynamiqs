from __future__ import annotations

from typing import get_args

from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..time_array import TimeArray, _factory_constant


def _astimearray(
    x: ArrayLike | TimeArray, dtype: jnp.complex64 | jnp.complex128
) -> TimeArray:
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype)
    else:
        return x
