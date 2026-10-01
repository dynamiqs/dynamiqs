"""A loop-free version of Diffrax's search of the next jump time.

`ClipStepSizeController` looks for the next jump time with two `while_loop`s (a linear
search up and down from the previous index). On GPU, each costs at least one
device-to-host synchronisation per step: XLA copies the loop condition back to the host
before each iteration. `JumpClipController` finds the same index by counting the jump
times before `t`, a single fused reduction over the (few) jump times.
"""

from __future__ import annotations

import diffrax as dx
import diffrax._step_size_controller.clip as dx_clip
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike


def _find_idx_by_counting(t: ArrayLike, ts: Array | None, hint: ArrayLike) -> ArrayLike:
    # index of the first element of the sorted `ts` strictly greater than `t`, as
    # Diffrax's `_find_idx_with_hint` (whose `hint` is only a starting point)
    if ts is None:
        return 0
    return jnp.sum(ts <= t, dtype=jnp.result_type(hint))


class JumpClipController(dx.ClipStepSizeController):
    """Diffrax's `ClipStepSizeController`, with a loop-free search of the next jump."""

    def __init__(
        self,
        controller: dx.AbstractAdaptiveStepSizeController,
        step_ts: ArrayLike | None = None,
        jump_ts: ArrayLike | None = None,
    ):
        super().__init__(controller, step_ts=step_ts, jump_ts=jump_ts)

    def adapt_step_size(self, *args, **kwargs) -> tuple:
        # swap Diffrax's search while this method is traced
        find_idx = dx_clip._find_idx_with_hint
        dx_clip._find_idx_with_hint = _find_idx_by_counting  # ty: ignore[invalid-assignment]
        try:
            return super().adapt_step_size(*args, **kwargs)
        finally:
            dx_clip._find_idx_with_hint = find_idx
