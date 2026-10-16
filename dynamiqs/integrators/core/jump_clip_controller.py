"""Diffrax's `ClipStepSizeController`, with two workarounds for Diffrax 0.7.2.

**Steps after a jump time move with it**, as in diffrax pull request #773. Without it,
the derivative with respect to close traced jump times is wrong by many orders of
magnitude (diffrax issue #772); `_drop_step_tangent` says why. Once the installed
Diffrax fixes it, the strict xfail `test_derivative_wrt_close_jump_times[diffrax]`
XPASSes, which fails the test suite: raise the `diffrax` lower bound in `pyproject.toml`
to that release, then remove `_drop_step_tangent`.

**The search of the next jump time is loop-free.** `ClipStepSizeController` looks for
the next jump time with two `while_loop`s (a linear search up and down from the previous
index). On GPU, each costs at least one device-to-host synchronisation per step: XLA
copies the loop condition back to the host before each iteration. `JumpClipController`
finds the same index by counting the jump times before `t`, a single fused reduction
over the (few) jump times.

It relies on a private function of Diffrax, `_find_idx_with_hint` in
`diffrax._step_size_controller.clip`, which it swaps while `adapt_step_size` is traced.
If a Diffrax release moves or renames it, `JumpClipController` falls back to Diffrax's
own search (the same steps, with the while loops), and
`test_jump_clip_controller.py::test_private_search_exists` fails, as a reminder.
Diffrax issue #776 tracks these loops, which are also in its search of the next
`step_ts` time. Once the installed Diffrax has none (e.g. with diffrax pull request
#777), the strict xfail `test_diffrax_search_is_loop_free` XPASSes, which fails the
test suite: raise the `diffrax` lower bound in `pyproject.toml` to that release, then
remove the swap (`_clip`, `_find_idx_by_counting`, `private_search_available` and their
tests), or, once `_drop_step_tangent` is gone too, use `dx.ClipStepSizeController` in
`diffrax_integrator.py` and delete this module and its tests.

`_drop_step_tangent` also reads the private state of `ClipStepSizeController`
(`step_info`, `jump_info` and `reject_info`): if a Diffrax release changes it,
`test_jump_clip_controller.py::test_private_clip_state` fails, as a reminder.
"""

from __future__ import annotations

import diffrax as dx
import equinox.internal as eqxi
import jax
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


def _get_t(i: ArrayLike, ts: Array) -> Array:
    # `ts[i]`, or `inf` past the end, as Diffrax's clip
    if len(ts) == 0:
        return jnp.asarray(jnp.inf)
    return jnp.where(i == len(ts), jnp.inf, ts[jnp.minimum(i, len(ts) - 1)])


def _drop_step_tangent(
    next_t0: Array,
    next_t1: Array,
    made_jump: Array,
    step_info: tuple[Array, Array] | None,
    jump_info: tuple[Array, Array] | None,
) -> Array:
    # A step clipped to just before a jump time is often tiny, but the tangent of its
    # length, d(jump) - d(t0), is O(1), and `PIDController` scales it into every
    # following step. So after a step that landed on a jump or `step_ts` time, give
    # the proposed `next_t1` the tangent of `next_t0`, keeping its value: the
    # following steps move with the time just landed on. This replays diffrax#773 on
    # the output of `adapt_step_size`, with the same values bit for bit. There, the
    # tangent is dropped before each clip: a `next_t1` clipped to the next jump time
    # keeps the clip's tangent, and so does one clipped to the next `step_ts` time,
    # unless the step landed on a jump (handled after the `step_ts` clip). It
    # assumes, as for every controller dynamiqs builds, that the inner controller
    # never reports `made_jump` and that rejected steps are not stored.
    move = jnp.asarray(False)
    if step_info is not None:
        # `index` is that of the first `step_ts` time after the step's end
        index, ts = step_info
        last = _get_t(index - 1, ts)
        landed = (index > 0) & ((last == next_t0) | (last == eqxi.nextafter(next_t0)))
        move = landed & (next_t1 != _get_t(index, ts))
    move = move | made_jump
    if jump_info is not None:
        index, ts = jump_info
        move = move & (next_t1 != eqxi.prevbefore(_get_t(index, ts)))
    moved_t1 = jax.lax.stop_gradient(next_t1) + (
        next_t0 - jax.lax.stop_gradient(next_t0)
    )
    return jnp.where(move, moved_t1, next_t1)


class JumpClipController(dx.ClipStepSizeController):
    """Diffrax's `ClipStepSizeController`, with steps that move with the jump and
    `step_ts` times they land on, and a loop-free search of the next jump.
    """

    def __init__(
        self,
        controller: dx.AbstractAdaptiveStepSizeController,
        step_ts: ArrayLike | None = None,
        jump_ts: ArrayLike | None = None,
    ):
        super().__init__(controller, step_ts=step_ts, jump_ts=jump_ts)

    def adapt_step_size(self, *args, **kwargs) -> tuple:
        keep_step, next_t0, next_t1, made_jump, state, result = self._clip(
            *args, **kwargs
        )
        assert state.reject_info is None  # see `_drop_step_tangent`
        next_t1 = _drop_step_tangent(
            next_t0, next_t1, made_jump, state.step_info, state.jump_info
        )
        return keep_step, next_t0, next_t1, made_jump, state, result

    def _clip(self, *args, **kwargs) -> tuple:
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
