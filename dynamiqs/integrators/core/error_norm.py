"""An error norm for Diffrax's `PIDController` that rejects failed complex steps.

Why: when a step fails, Diffrax sets its error to `inf`, that is `inf + 0j` for a
complex state. `PIDController` divides it by a real scale promoted to complex, which
gives `inf + nanj` and a NaN norm. The next step size is then NaN, and the solve runs
until `max_steps` (diffrax issue #774, Diffrax up to at least 0.7.2). In Diffrax 0.7.2,
`sesolve` and `mesolve` of a driven qubit with `Kvaerno5` hit this.

What: `error_norm` returns `inf` instead of a NaN norm, so the failed step is rejected
and retried with a smaller step, as for a real state. Otherwise it returns Diffrax's
norm unchanged. The root finder of the implicit methods (`VeryChord`) uses this norm
too, and treats `inf` like NaN: as diverged.

To remove: once Diffrax fixes the division (e.g. diffrax pull request #775), the strict
xfail `test_error_norm.py::test_complex_failed_step_is_rejected[diffrax]` passes and
fails the test suite. Raise the `diffrax` lower bound in `pyproject.toml` to that
release, then remove `error_norm`.
"""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jaxtyping import PyTree


def error_norm(x: PyTree) -> Array:
    """Diffrax's default norm, `optx.rms_norm`, with `inf` in place of NaN."""
    norm = optx.rms_norm(x)
    return jnp.where(jnp.isnan(norm), jnp.inf, norm)
