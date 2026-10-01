"""A loop-free version of Diffrax's search of the next jump time.

`ClipStepSizeController` looks for the next jump time with two `while_loop`s (a linear
search up and down from the previous index). On GPU, each costs at least one
device-to-host synchronisation per step: XLA copies the loop condition back to the host
before each iteration. `JumpClipController` finds the same index by counting the jump
times before `t`, a single fused reduction over the (few) jump times.

It relies on a private function of Diffrax, `_find_idx_with_hint` in
`diffrax._step_size_controller.clip`, which it swaps while `adapt_step_size` is traced.
If a Diffrax release moves or renames it, `JumpClipController` falls back to Diffrax's
own search (the same steps, with the while loops), and
`test_jump_clip_controller.py::test_private_search_exists` fails, as a reminder.
"""

from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

try:
    import diffrax._step_size_controller.clip as dx_clip
except ImportError:  # moved in a newer Diffrax: fall back to its search
    dx_clip = None


def private_search_available() -> bool:
    """Whether Diffrax's private `_find_idx_with_hint`, which `JumpClipController`
    replaces, is where it expects it.
    """
    return dx_clip is not None and callable(
        getattr(dx_clip, '_find_idx_with_hint', None)
    )


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
        clip = dx_clip
        if clip is None or not private_search_available():
            return super().adapt_step_size(*args, **kwargs)
        # swap Diffrax's search while this method is traced
        find_idx = clip._find_idx_with_hint
        clip._find_idx_with_hint = _find_idx_by_counting  # ty: ignore[invalid-assignment]
        try:
            return super().adapt_step_size(*args, **kwargs)
        finally:
            clip._find_idx_with_hint = find_idx
