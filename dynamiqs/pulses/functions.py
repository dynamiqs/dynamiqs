from __future__ import annotations

from jax import Array
from jax.tree_util import Partial
import jax.numpy as jnp
from .envelopes import (
    format_pulse_param,
    flat,
    raised_cosine_envelope,
    raised_cosine_drag_envelope,
)
from .filters import prepare_gaussian_params, gaussian_filter_closure_func

__all__ = ["flat_top_gaussian", "raised_cosine", "raised_cosine_gaussian_filtered"]


def flat_top_gaussian(
    pixel_times: Array,
    hold_amplitudes: float | Array,
    pad_times: float | Array,
    hold_times: float | Array,
    gaussian_std: float | Array,
) -> callable[[float], Array]:
    hold_amplitudes = format_pulse_param(hold_amplitudes)
    pixel_amplitudes = flat(pixel_times, pad_times=pad_times, hold_times=hold_times)
    # Shape (Npix, pad_dim, hold_dim, amp_dim)
    pixel_amplitudes = pixel_amplitudes[..., None] * hold_amplitudes
    # Shape (sigma?, pad_dim?, hold_dim?, amp_dim?)
    pixel_times, pixel_sizes, timescale = prepare_gaussian_params(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )

    # Shape (batch_filter?, batch_pad?, batch_hold?, batch_amp?)
    return Partial(
        gaussian_filter_closure_func,
        pixel_times=pixel_times,
        pixel_sizes=pixel_sizes,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )


def raised_cosine(
    amplitudes: float | Array,
    gate_times: float | Array,
    carrier_freqs: float | Array,
    carrier_phases: float | Array,
) -> callable[[float], Array]:
    amplitudes = format_pulse_param(amplitudes)
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)

    def closure_func(
        t: float,
        amplitudes: float | Array,
        gate_times: float | Array,
        carrier_freqs: float | Array,
        carrier_phases: float | Array,
    ) -> callable:
        # Shape (time_dim?, freq_dim?, phase_dim?)
        pixel_envelopes = raised_cosine_envelope(
            t,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
        )
        # Shape (time_dim?, freq_dim?, phase_dim?, amp_dim?)
        return jnp.squeeze(pixel_envelopes[..., None] * amplitudes)

    return Partial(
        closure_func,
        amplitudes=amplitudes,
        gate_times=gate_times,
        carrier_freqs=carrier_freqs,
        carrier_phases=carrier_phases,
    )


def raised_cosine_gaussian_filtered(
    pixel_times: Array,
    amplitudes: float | Array,
    gate_times: float | Array,
    carrier_freqs: float | Array,
    carrier_phases: float | Array,
    gaussian_std: float | Array,
) -> callable[[float], Array]:
    amplitudes = format_pulse_param(amplitudes)
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)
    pixel_amplitudes = raised_cosine_envelope(
        pixel_times,
        gate_times=gate_times,
        carrier_freqs=carrier_freqs,
        carrier_phases=carrier_phases,
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?, amp_dim)
    pixel_amplitudes = pixel_amplitudes[..., None] * amplitudes
    pixel_times, pixel_sizes, timescale = prepare_gaussian_params(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )
    # Shape (batch_filter?, batch_time?, batch_freq?, batch_phase?, batch_amp?)
    return Partial(
        gaussian_filter_closure_func,
        pixel_times=pixel_times,
        pixel_sizes=pixel_sizes,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )
