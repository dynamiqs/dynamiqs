from __future__ import annotations

from typing import Any

import jax.numpy as jnp


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


def extract_nsteps(result: Any) -> float | None:
    infos = getattr(result, 'infos', None)
    if infos is None:
        return None
    nsteps = getattr(infos, 'nsteps', None)
    if nsteps is None:
        return None
    nsteps = jnp.asarray(nsteps)
    if nsteps.size == 0:
        return None
    return float(jnp.mean(nsteps))
