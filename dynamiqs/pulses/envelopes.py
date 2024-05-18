from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

__all__ = ["format_pulse_param", "flat_envelope", "raised_cosine", "raised_cosine_drag"]


def format_pulse_param(parameter: float | ArrayLike) -> ArrayLike:
    if type(parameter) is not float and len(parameter.shape) > 1:
        raise ValueError(
            f"Parameter needs to be a 1D array or a float, but got shape {parameter.shape}"
        )
    return jnp.atleast_1d(parameter)


def flat(
    t: float | Array, /, pad_times: float | Array, hold_times: float | Array
) -> Array:
    t = format_pulse_param(t).reshape(len(t), 1, 1)
    pad_times = format_pulse_param(pad_times).reshape(1, len(pad_times), 1)
    hold_times = format_pulse_param(hold_times).reshape(1, 1, len(hold_times))
    nonzero_times = (t > pad_times) & (t <= pad_times + hold_times)
    return jnp.squeeze(jnp.where(nonzero_times, 1, 0))


def raised_cosine(
    t: float | Array,
    /,
    gate_times_ns: float | Array,
    carrier_freqs_GHz: float | Array,
    carrier_phases: float | Array = 0.0,
) -> Array:
    """
    raised_cosine_pulse A simple cosine envelope that starts and end at 0 amplitude at t=0.0
    and t=gate_time_ns, and reaches a maximal amplitude of `1.0`.

    Note: The mean amplitude is `1/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time_ns` and `amplitude_GHz = amplitude / (2 * pi)`.

    Args:
        t (float): Time, in ns.
        gate_times_ns (float): _description_
        carrier_freqs_GHz (float): _description_
        carrier_phases (float, optional): _description_. Defaults to 0.0.

    Returns:
        float: Shape(
            len(t)?,
            len(gate_times_ns)?,
            len(carrier_freqs_GHz)?,
            len(carrier_phases)?
        )
    """
    t = format_pulse_param(t).reshape(len(t), 1, 1, 1)
    gate_times_ns = format_pulse_param(gate_times_ns).reshape(
        1, len(gate_times_ns), 1, 1
    )
    carrier_freqs_GHz = format_pulse_param(carrier_freqs_GHz).reshape(
        1, 1, len(carrier_freqs_GHz), 1
    )
    carrier_phases = format_pulse_param(carrier_phases).reshape(
        1, 1, 1, len(carrier_phases)
    )
    return jnp.squeeze(
        (1 - jnp.cos(2 * jnp.pi / gate_times_ns * t))
        / 2
        * jnp.cos(carrier_freqs_GHz * t + carrier_phases)
    )


def raised_cosine_drag(
    t: float | Array,
    /,
    gate_times_ns: float | Array,
    carrier_freqs_GHz: float | Array,
    carrier_phases: float | Array = 0.0,
    drag_params: float | Array = 0.0,
) -> Array:
    """
    raised_cosine_drag A simple cosine envelope with "quadrature" DRAG pulse that starts
    and end at 0 at t=0.0 and t=gate_time_ns, and reaches a maximal amplitude of `1.0`.

    Note: The mean amplitude is `1/2` such that to implement a rotation of
        angle theta, the prefactor in front of sigma_j/2 in the Hamiltonian should be
        `amplitude = theta / gate_time_ns` and `amplitude_GHz = amplitude / (2 * pi)`.

    Args:
        t (float | Array): _description_
        gate_times_ns (float | Array): _description_
        carrier_freqs_GHz (float | Array): _description_
        carrier_phases (float | Array, optional): _description_. Defaults to 0.0.
        drag_params (float | Array, optional): _description_. Defaults to 0.0.

    Returns:
        Array: Shape(
            len(t)?,
            len(gate_times_ns)?,
            len(carrier_freqs_GHz)?,
            len(carrier_phases)?,
            len(drag_params)?
        )
    """
    t = format_pulse_param(t).reshape(len(t), 1, 1, 1, 1)
    gate_times_ns = format_pulse_param(gate_times_ns).reshape(
        1, len(gate_times_ns), 1, 1, 1
    )
    carrier_freqs_GHz = format_pulse_param(carrier_freqs_GHz).reshape(
        1, 1, len(carrier_freqs_GHz), 1, 1
    )
    carrier_phases = format_pulse_param(carrier_phases).reshape(
        1, 1, 1, len(carrier_phases), 1
    )
    drag_params = format_pulse_param(drag_params).reshape(1, 1, 1, 1, len(drag_params))
    in_phase_envelope = (
        (1 - jnp.cos(2 * jnp.pi / gate_times_ns * t))
        / 2
        * jnp.cos(carrier_freqs_GHz * t + carrier_phases)
    )
    quadrature_envelope = (
        drag_params
        * jnp.sin(2 * jnp.pi / gate_times_ns * t)
        * (jnp.pi / gate_times_ns)
        * jnp.sin(carrier_freqs_GHz * t + carrier_phases)
    )
    return jnp.squeeze(in_phase_envelope + quadrature_envelope)
