from __future__ import annotations

from jax import Array
from jax.tree_util import Partial
import jax.numpy as jnp
from .envelopes import (
    format_pulse_params,
    _flat,
    _raised_cosine_envelope,
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
    # hold_amplitudes = format_pulse_param(hold_amplitudes)
    # pixel_amplitudes = flat(pixel_times, pad_times=pad_times, hold_times=hold_times)
    # # Shape (Npix, pad_dim, hold_dim, amp_dim)
    # pixel_amplitudes = pixel_amplitudes[..., None] * hold_amplitudes
    # # Shape (sigma?, pad_dim?, hold_dim?, amp_dim?)

    pixel_times_expanded, hold_amplitudes, pad_times, hold_times = format_pulse_params(
        [pixel_times, hold_amplitudes, pad_times, hold_times]
    )
    # Shape (pixel_dim, amp_dim?, sigma?, pad_dim?, hold_dim?)
    pixel_amplitudes = jnp.squeeze(
        hold_amplitudes
        * _flat(pixel_times_expanded, pad_times=pad_times, hold_times=hold_times)
    )

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
    amplitudes, gate_times, carrier_freqs, carrier_phases = format_pulse_params(
        (amplitudes, gate_times, carrier_freqs, carrier_phases)
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)

    def closure_func(
        t: float,
        amplitudes: Array,
        gate_times: Array,
        carrier_freqs: Array,
        carrier_phases: Array,
    ) -> callable:
        # Shape (amp_dim?, time_dim?, freq_dim?, phase_dim?)
        return jnp.squeeze(
            amplitudes
            * _raised_cosine_envelope(
                t,
                gate_times=gate_times,
                carrier_freqs=carrier_freqs,
                carrier_phases=carrier_phases,
            )
        )

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
    pixel_times_expanded, amplitudes, gate_times, carrier_freqs, carrier_phases = (
        format_pulse_params(
            (pixel_times, amplitudes, gate_times, carrier_freqs, carrier_phases)
        )
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?)
    pixel_amplitudes = jnp.squeeze(
        amplitudes
        * _raised_cosine_envelope(
            pixel_times_expanded,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
        )
    )
    # Shape (Npix, time_dim?, freq_dim?, phase_dim?, amp_dim)
    pixel_times, pixel_sizes, timescale = prepare_gaussian_params(
        pixel_times=pixel_times,
        pixel_amplitudes=pixel_amplitudes,
        gaussian_std=gaussian_std,
    )
    # Shape (batch_filter?, batch_amp?, batch_time?, batch_freq?, batch_phase?)
    return Partial(
        gaussian_filter_closure_func,
        pixel_times=pixel_times,
        pixel_sizes=pixel_sizes,
        pixel_amplitudes=pixel_amplitudes,
        timescale=timescale,
    )
