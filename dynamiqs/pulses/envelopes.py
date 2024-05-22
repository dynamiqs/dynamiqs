from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Iterable

__all__ = [
    "format_pulse_param",
    "format_pulse_params",
    "flat",
    "raised_cosine_envelope",
    "raised_cosine_drag_envelope",
]

FloatOrComplex = float | complex
PulseParamType = float | complex | Array


def format_pulse_param(parameter: PulseParamType) -> Array:
    if (
        type(parameter) is not float
        and type(parameter) is not complex
        and len(parameter.shape) > 1
    ):
        raise ValueError(
            f"Parameter needs to be a 1D array or a float, but got shape {parameter.shape}"
        )
    return jnp.atleast_1d(parameter)


def format_pulse_params(parameters: Iterable[PulseParamType]) -> list[Array]:
    # Transforms N parameters into unsqueezed shape (len(param), 1, ..., 1) for explicit
    # vectorization. Keep the sequential order of the parameters.
    # E.g. with parameters = [jnp.arange(3), 1.2, jnp.arange(7)], the return formated
    # parameters will be shapes (3, 1, 1), (1, 1, 1), and (1, 1, 7), respectively.
    parameters = [format_pulse_param(param) for param in parameters]
    N_params = len(parameters)
    return [
        jnp.expand_dims(param, list(range(0, idx)) + list(range(idx + 1, N_params)))
        for idx, param in enumerate(parameters)
    ]


def _flat(t: float | Array, /, pad_times: Array, hold_times: Array) -> Array:
    nonzero_times = (t > pad_times) & (t <= pad_times + hold_times)
    return jnp.where(nonzero_times, 1, 0)


def flat(
    t: float | Array, /, pad_times: PulseParamType, hold_times: PulseParamType
) -> Array:
    t, pad_times, hold_times = format_pulse_params([t, pad_times, hold_times])
    return jnp.squeeze(_flat(t, pad_times=pad_times, hold_times=hold_times))


def _raised_cosine_envelope(
    t: float | Array,
    /,
    gate_times: Array,
    carrier_freqs: Array,
    carrier_phases: Array,
) -> Array:
    return (
        (1 - jnp.cos(2 * jnp.pi / gate_times * t))
        / 2
        * jnp.cos(carrier_freqs * t + carrier_phases)
    )


def raised_cosine_envelope(
    t: float | Array,
    /,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType,
    carrier_phases: PulseParamType = 0.0,
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
    t, gate_times, carrier_freqs, carrier_phases = format_pulse_params(
        [t, gate_times, carrier_freqs, carrier_phases]
    )
    return jnp.squeeze(
        _raised_cosine_envelope(
            t,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
        )
    )


def _raised_cosine_drag_envelope(
    t: float | Array,
    /,
    gate_times: Array,
    carrier_freqs: Array,
    carrier_phases: Array,
    drag_params: Array,
) -> Array:
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
    return in_phase_envelope + quadrature_envelope


def raised_cosine_drag_envelope(
    t: float | Array,
    /,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType,
    carrier_phases: PulseParamType = 0.0,
    drag_params: PulseParamType = 0.0,
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
    t, gate_times, carrier_freqs, carrier_phases, drag_params = format_pulse_params(
        [t, gate_times, carrier_freqs, carrier_phases, drag_params]
    )
    return jnp.squeeze(
        _raised_cosine_drag_envelope(
            t,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
            drag_params=drag_params,
        )
    )
