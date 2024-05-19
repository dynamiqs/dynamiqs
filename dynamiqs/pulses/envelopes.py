from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike
from typing import Iterable

__all__ = [
    "format_pulse_param",
    "flat",
    "raised_cosine_envelope",
    "raised_cosine_drag_envelope",
]


def format_pulse_param(parameter: float | ArrayLike) -> ArrayLike:
    if type(parameter) is not float and len(parameter.shape) > 1:
        raise ValueError(
            f"Parameter needs to be a 1D array or a float, but got shape {parameter.shape}"
        )
    return jnp.atleast_1d(parameter)


def flat(
    t: float | Array, /, pad_times: float | Array, hold_times: float | Array
) -> Array:
    t = format_pulse_param(t)
    pad_times = format_pulse_param(pad_times)
    hold_times = format_pulse_param(hold_times)
    t = t.reshape(len(t), 1, 1)
    pad_times = pad_times.reshape(1, len(pad_times), 1)
    hold_times = hold_times.reshape(1, 1, len(hold_times))
    nonzero_times = (t > pad_times) & (t <= pad_times + hold_times)
    return jnp.squeeze(jnp.where(nonzero_times, 1, 0))


def _raised_cosine_envelope(
    t: float | Array,
    /,
    gate_times: Array,
    carrier_freqs: Array,
    carrier_phases: Array,
) -> Array:
    return jnp.squeeze(
        (1 - jnp.cos(2 * jnp.pi / gate_times * t))
        / 2
        * jnp.cos(carrier_freqs * t + carrier_phases)
    )


def raised_cosine_envelope(
    t: float | Array,
    /,
    gate_times: float | Array,
    carrier_freqs: float | Array,
    carrier_phases: float | Array = 0.0,
) -> Array:
    """
    raised_cosine_pulse A simple cosine envelope that starts and end at 0 amplitude at t=0.0
    and t=gate_time, and reaches a maximal amplitude of `1.0`.

    Note: The mean amplitude is `1/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time` and `amplitude = amplitude / (2 * pi)`.

    Args:
        t (float): Time, in ns.
        gate_times (float): _description_
        carrier_freqs (float): _description_
        carrier_phases (float, optional): _description_. Defaults to 0.0.

    Returns:
        float: Shape(
            len(t)?,
            len(gate_times)?,
            len(carrier_freqs)?,
            len(carrier_phases)?
        )
    """
    t = format_pulse_param(t)
    gate_times = format_pulse_param(gate_times)
    carrier_freqs = format_pulse_param(carrier_freqs)
    carrier_phases = format_pulse_param(carrier_phases)
    t = t.reshape(len(t), 1, 1, 1)
    gate_times = gate_times.reshape(1, len(gate_times), 1, 1)
    carrier_freqs = carrier_freqs.reshape(1, 1, len(carrier_freqs), 1)
    carrier_phases = carrier_phases.reshape(1, 1, 1, len(carrier_phases))
    # return jnp.squeeze(
    #     (1 - jnp.cos(2 * jnp.pi / gate_times * t))
    #     / 2
    #     * jnp.cos(carrier_freqs * t + carrier_phases)
    # )
    return _raised_cosine_envelope(
        t,
        gate_times=gate_times,
        carrier_freqs=carrier_freqs,
        carrier_phases=carrier_phases,
    )


def raised_cosine_drag_envelope(
    t: float | Array,
    /,
    gate_times: float | Array,
    carrier_freqs: float | Array,
    carrier_phases: float | Array = 0.0,
    drag_params: float | Array = 0.0,
) -> Array:
    """
    raised_cosine_drag A simple cosine envelope with "quadrature" DRAG pulse that starts
    and end at 0 at t=0.0 and t=gate_time, and reaches a maximal amplitude of `1.0`.

    Note: The mean amplitude is `1/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time` and `amplitude = amplitude / (2 * pi)`.

    Args:
        t (float | Array): _description_
        gate_times (float | Array): _description_
        carrier_freqs (float | Array): _description_
        carrier_phases (float | Array, optional): _description_. Defaults to 0.0.
        drag_params (float | Array, optional): _description_. Defaults to 0.0.

    Returns:
        Array: Shape(
            len(t)?,
            len(gate_times)?,
            len(carrier_freqs)?,
            len(carrier_phases)?,
            len(drag_params)?
        )
    """
    t = format_pulse_param(t)
    gate_times = format_pulse_param(gate_times)
    carrier_freqs = format_pulse_param(carrier_freqs)
    carrier_phases = format_pulse_param(carrier_phases)
    drag_params = format_pulse_param(drag_params)
    t = t.reshape(len(t), 1, 1, 1, 1)
    gate_times = gate_times.reshape(1, len(gate_times), 1, 1, 1)
    carrier_freqs = carrier_freqs.reshape(1, 1, len(carrier_freqs), 1, 1)
    carrier_phases = carrier_phases.reshape(1, 1, 1, len(carrier_phases), 1)
    drag_params = drag_params.reshape(1, 1, 1, 1, len(drag_params))
    in_phase_envelope = (
        (1 - jnp.cos(2 * jnp.pi / gate_times * t))
        / 2
        * jnp.cos(carrier_freqs * t + carrier_phases)
    )
    quadrature_envelope = (
        drag_params
        * jnp.sin(2 * jnp.pi / gate_times * t)
        * (jnp.pi / gate_times)
        * jnp.sin(carrier_freqs * t + carrier_phases)
    )
    return jnp.squeeze(in_phase_envelope + quadrature_envelope)
