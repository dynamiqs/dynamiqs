from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

import dynamiqs as dq


@dataclass(frozen=True)
class MetricStats:
    mean: float
    minimum: float
    maximum: float


def aggregate_stats(values: Any) -> MetricStats:
    values = jnp.asarray(values)
    return MetricStats(
        mean=float(jnp.mean(values)),
        minimum=float(jnp.min(values)),
        maximum=float(jnp.max(values)),
    )


def relative_l2_error(value: Any, reference: Any) -> float:
    value = jnp.asarray(value)
    reference = jnp.asarray(reference)
    numerator = jnp.linalg.norm(value - reference)
    denominator = jnp.maximum(jnp.linalg.norm(reference), 1e-14)
    return float(numerator / denominator)


def max_abs_error(value: Any, reference: Any) -> float:
    value = jnp.asarray(value)
    reference = jnp.asarray(reference)
    return float(jnp.max(jnp.abs(value - reference)))


def state_infidelity_stats(states: Any, reference_states: Any) -> MetricStats:
    """Return batch statistics of the mean trajectory infidelity."""
    fidelities = jnp.clip(dq.fidelity(states, reference_states), 0.0, 1.0)
    infidelities = 1.0 - fidelities
    mean_time_infidelity = jnp.mean(infidelities, axis=-1)
    return aggregate_stats(mean_time_infidelity)


def extract_nsteps_stats(result: Any) -> MetricStats | None:
    infos = getattr(result, 'infos', None)
    if infos is None:
        return None
    nsteps = getattr(infos, 'nsteps', None)
    if nsteps is None:
        return None
    nsteps = jnp.asarray(nsteps)
    if nsteps.size == 0:
        return None
    return aggregate_stats(nsteps)


def extract_nsteps(result: Any) -> float | None:
    """Return the mean step count, preserving the previous helper API."""
    stats = extract_nsteps_stats(result)
    return None if stats is None else stats.mean
