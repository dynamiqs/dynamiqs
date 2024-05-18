from __future__ import annotations

from jax import Array
from jax.tree_util import Partial
from .envelopes import format_pulse_param, flat_envelope
from .filters import prepare_gaussian_params, gaussian_filter_closure_func

__all__ = ["flat_top_gaussian"]


def flat_top_gaussian(
    pixel_times: Array,
    hold_amplitudes: float | Array,
    pad_times: float | Array,
    hold_times: float | Array,
    gaussian_std: float | Array,
    # times_defined_at_mid_pix: bool = False,
) -> callable:
    hold_amplitudes = format_pulse_param(hold_amplitudes)
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
    return Partial(
        gaussian_filter_closure_func,
        pixel_times=pixel_times,
        pixel_sizes=pixel_sizes,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )
