from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf
from jaxtyping import ArrayLike


__all__ = [
    "prepare_gaussian_params",
    "gaussian_filter_closure_func",
]


def prepare_gaussian_params(
    pixel_times: ArrayLike,
    pixel_amplitudes: float | ArrayLike,
    gaussian_std: float | ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    pixel_sizes = jnp.diff(pixel_times)
    pixel_sizes = jnp.insert(pixel_sizes, -1, pixel_sizes[-1])[:, None]
    if len(pixel_times) == len(jnp.atleast_1d(pixel_amplitudes)):
        pixel_times = jnp.insert(pixel_times, -1, pixel_times[-1])[:, None]
        pixel_sizes = jnp.insert(pixel_sizes, -1, pixel_sizes[-1])[:, None]
    else:
        pixel_times = pixel_times[:, None]
    timescale = 1 / (jnp.sqrt(2) * jnp.atleast_1d(gaussian_std))[None]  # (1, Nsig)
    return pixel_times, pixel_sizes, timescale


def gaussian_filter_closure_func(
    t: float,
    pixel_times: ArrayLike,
    pixel_sizes: float | ArrayLike,
    pixel_amplitudes: ArrayLike,
    timescale: ArrayLike,
) -> ArrayLike:
    erfs = -0.5 * jnp.diff(
        erf((t - (pixel_times - pixel_sizes / 2)) * timescale), axis=0
    )
    erfs = erfs / jnp.sum(erfs, axis=0)
    output_amps = jnp.einsum("ps,p...->s...", erfs, pixel_amplitudes)
    return jnp.squeeze(output_amps)
