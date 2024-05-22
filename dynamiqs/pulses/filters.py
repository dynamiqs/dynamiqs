from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf
from jax import Array
from .envelopes import (
    PulseParamType,
)


__all__ = [
    "prepare_gaussian_params",
    "gaussian_filter_closure_func",
]


def prepare_gaussian_params(
    pixel_times: Array,
    pixel_amplitudes: PulseParamType,
    gaussian_std: PulseParamType,
) -> tuple[Array, Array, Array]:
    pixel_sizes = jnp.diff(pixel_times)
    # pixel_sizes = jnp.insert(pixel_sizes, -1, pixel_sizes[-1])
    pixel_sizes = jnp.concatenate([pixel_sizes, pixel_sizes[-1:]], axis=0)
    if len(pixel_times) == len(jnp.atleast_1d(pixel_amplitudes)):
        pixel_times = jnp.concatenate(
            [pixel_times, pixel_times[-1:] + pixel_sizes[-1]], axis=0
        )[:, None]
        pixel_sizes = jnp.concatenate([pixel_sizes, pixel_sizes[-1:]], axis=0)[:, None]
    else:
        pixel_times = pixel_times[:, None]
        pixel_sizes = pixel_sizes[:, None]
    timescale = 1 / (jnp.sqrt(2) * jnp.atleast_1d(gaussian_std))[None]  # (1, Nsig)
    return pixel_times, pixel_sizes, timescale


def gaussian_filter_closure_func(
    t: float,
    pixel_times: Array,
    pixel_sizes: PulseParamType,
    pixel_amplitudes: Array,
    timescale: Array,
) -> Array:
    erfs = -0.5 * jnp.diff(
        erf((t - (pixel_times - pixel_sizes / 2)) * timescale), axis=0
    )
    erfs = erfs / jnp.sum(erfs, axis=0)
    output_amps = jnp.einsum("ps,p...->s...", erfs, pixel_amplitudes)
    return jnp.squeeze(output_amps)
