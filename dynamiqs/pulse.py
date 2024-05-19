from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array, lax
from jax.scipy.special import erf
from jaxtyping import ArrayLike, PyTree, ScalarLike

__all__ = ["flat_top_gaussian"]


def format_pulse_param(parameter: float | ArrayLike) -> ArrayLike:
    if type(parameter) is not float and len(parameter.shape) > 1:
        raise ValueError(
            f"Parameter needs to be a 1D array or a float, but got shape {parameter.shape}"
        )
    return jnp.atleast_1d(parameter)


def flat_envelope(
    t: float | Array, /, pad_times: float | Array, hold_times: float | Array
) -> Array:
    t = jnp.atleast_1d(t).reshape(len(t), 1, 1)
    pad_times = jnp.atleast_1d(pad_times).reshape(1, len(pad_times), 1)
    hold_times = jnp.atleast_1d(hold_times).reshape(1, 1, len(hold_times))
    nonzero_times = (t > pad_times) & (t <= pad_times + hold_times)
    envelope = jnp.where(nonzero_times, 1, 0)
    return envelope


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


def flat_top_gaussian(
    pixel_times: Array,
    hold_amplitudes: float | Array,
    pad_times: float | Array,
    hold_times: float | Array,
    gaussian_std: float | Array,
    # times_defined_at_mid_pix: bool = False,
) -> callable:
    hold_amplitudes = jnp.atleast_1d(hold_amplitudes)
    pixel_amplitudes = flat_envelope(
        pixel_times, pad_times=pad_times, hold_times=hold_times
    )
    # Shape (Npix, pad_dim, hold_dim, amp_dim)
    pixel_amplitudes = pixel_amplitudes[..., None] * hold_amplitudes
    # Shape (sigma?, pad_dim?, hold_dim?, amp_dim?)
    pixel_times, pixel_sizes, timescale = prepare_gaussian_params(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )

    # Shape (batch_amplitude?, batch_start?, batch_hold?, batch_end?, batch_filter?)
    return jax.tree_util.Partial(
        gaussian_filter_closure_func,
        pixel_times=pixel_times,
        pixel_sizes=pixel_sizes,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )


# def apply_gaussian_filter(
#     t: float,
#     pixel_times: ArrayLike,
#     pixel_amplitudes: ArrayLike,
#     gaussian_std: float | ArrayLike,
# ) -> ArrayLike:
#     """
#     Potentially vectorizing over `gaussian_std` as well.
#     """
#     # Npix = len(pixel_times)
#     # pixel_times: shape (Npix,)
#     # pixel_amplitudes: shape (Npix, ...)
#     # gaussian_std: shape (Nsig) or float
#     # return: shape (Nsig?, ...)
#     pixel_times = jnp.insert(pixel_times, -1, pixel_times[-1])[:, None]  # (Npix, 1)
#     timescale = 1 / (jnp.sqrt(2) * jnp.atleast_1d(gaussian_std))[None]  # (1, Nsig)
#     # erfs = -0.5 * jnp.diff(erf((t - pixel_times) * timescale), axis=0)  # (Npix, Nsig)
#     pixel_size = pixel_times[1] - pixel_times[0]
#     erfs = -0.5 * jnp.diff(
#         erf((t - (pixel_times - pixel_size / 2)) * timescale), axis=0
#     )  # (Npix, Nsig)
#     erfs = erfs / jnp.sum(erfs)
#     # Vectorize for einsum
#     output_amps = jnp.einsum("ps,p...->s...", erfs, pixel_amplitudes)
#     return jnp.squeeze(output_amps)


def flat_top_gaussian(
    amplitude: float | ArrayLike,
    hold_time: float | ArrayLike,
    buffer_start: float | ArrayLike,
    buffer_end: float | ArrayLike,
    gaussian_filter_sigma: float | ArrayLike,
) -> callable:
    amplitude = jnp.atleast_1d(amplitude)
    hold_time = jnp.atleast_1d(hold_time)
    buffer_start = jnp.atleast_1d(buffer_start)
    buffer_end = jnp.atleast_1d(buffer_end)
    gaussian_filter_sigma = jnp.atleast_1d(gaussian_filter_sigma)

    buffer_start, hold_time, buffer_end = jnp.meshgrid(
        buffer_start, hold_time, buffer_end, indexing="ij"
    )
    lengths = jnp.stack(
        [jnp.zeros_like(buffer_start), buffer_start, hold_time, buffer_end], axis=0
    )
    times_drives = jnp.cumsum(lengths, axis=0)
    t_dim, s_dim, h_dim, e_dim = times_drives.shape
    f_dim = len(gaussian_filter_sigma)
    # Shape (4, start, hold, end), where dims are 1 if not batched over.
    times_drives = times_drives.reshape(t_dim, 1, s_dim, h_dim, e_dim, 1)
    timescale = 1 / (jnp.sqrt(2) * gaussian_filter_sigma).reshape(1, 1, 1, 1, 1, f_dim)

    a_dim = len(amplitude)
    zeros = jnp.zeros_like(amplitude)
    amplitudes = jnp.stack([zeros, amplitude, zeros], axis=0)  # (3, a_dim)
    amplitudes = amplitudes.reshape(3, a_dim, 1, 1, 1, 1)

    def f(
        t: float,
        times_drives: ArrayLike,
        timescale: ArrayLike,
        amplitudes: ArrayLike,
    ) -> ArrayLike:
        erfs = -0.5 * jnp.diff(erf((t - times_drives) * timescale), axis=0)
        erfs = erfs / jnp.sum(erfs)
        return jnp.squeeze((erfs * amplitudes).sum(axis=0))

    # Shape (batch_amplitude?, batch_start?, batch_hold?, batch_end?, batch_filter?)
    return jax.tree_util.Partial(
        f, times_drives=times_drives, timescale=timescale, amplitudes=amplitudes
    )
