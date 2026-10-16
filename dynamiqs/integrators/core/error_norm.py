"""An error norm for Diffrax's `PIDController` that rejects failed complex steps.

When a step fails (the root find of an implicit step fails, or the step is so large
that its error estimate is NaN), Diffrax reports `y_error = inf`, that is `inf + 0j`
for a complex `y`. `PIDController` divides it by the real scale
`atol + rtol * max(|y0|, |y1|)`, promoted to complex: the complex division gives
`inf + nanj`, whose norm is NaN. The next step size is then NaN, and the solve runs
until `max_steps` (diffrax issue #774, in Diffrax up to at least 0.7.2). This hits any
method on a complex state; with the implicit methods (`Kvaerno3`, `Kvaerno5`), any
step whose root find fails: in Diffrax 0.7.2, `sesolve` and `mesolve` of a driven
qubit with `Kvaerno5`.

`error_norm` returns `inf` instead of a NaN norm, so that the failed step is rejected
and retried with a smaller step, as for a real `y`. It returns Diffrax's norm unchanged
otherwise. The implicit methods' root finder (`VeryChord`) takes the controller's norm
too; it treats a NaN and an `inf` norm alike, as diverged.

Once a Diffrax release fixes the division (e.g. diffrax pull request #775),
the strict xfail `test_error_norm.py::test_complex_failed_step_is_rejected[diffrax]`
XPASSes, which fails the test suite: raise the `diffrax` lower bound in
`pyproject.toml` to that release, then remove `error_norm`.
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
