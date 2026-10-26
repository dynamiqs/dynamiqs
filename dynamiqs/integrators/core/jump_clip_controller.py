"""Diffrax's `ClipStepSizeController`, and three workarounds for Diffrax 0.7.2.

1. **Steps after a jump time move with it** (diffrax issue #772, fixed by diffrax pull
   request #773). Without this, derivatives with respect to close jump times are wrong
   by orders of magnitude. See `_drop_step_tangent`.
   To remove: once Diffrax fixes it, the strict xfail
   `test_derivative_wrt_close_jump_times[diffrax]` passes and fails the test suite.
   Raise the `diffrax` lower bound in `pyproject.toml` to that release, then remove
   `_drop_step_tangent`.

2. **The next jump time is found without a loop** (diffrax issue #776, fixed by diffrax
   pull request #777). Diffrax searches it with two `while_loop`s. On GPU, each loop
   iteration copies its condition to the host. `JumpClipController` counts the jump
   times before `t` instead, in one reduction. To do so, it swaps Diffrax's private
   `_find_idx_with_hint` while `adapt_step_size` is traced.
   To remove: once Diffrax has no loops, the strict xfail
   `test_diffrax_search_is_loop_free` passes and fails the test suite. Raise the
   `diffrax` lower bound, then remove the swap (`_clip`, `_find_idx_by_counting`,
   `private_search_available` and their tests).

3. **After a rejected step near t1, keep the controller's shorter step** (diffrax
   issue #756, fixed by diffrax pull request #758, not released yet). Diffrax moves
   a step that ends less than 100 ULP (units in the last place) before t1 onto t1.
   When an operator jumps in these
   last 100 ULP, that step crosses the jump and is rejected. The controller then
   proposes a shorter step, but Diffrax replaces it with a step that ends halfway to
   t1. That step crosses the jump too, so it is rejected again, and so on until
   `max_steps`. `keep_rejected_steps` swaps Diffrax's private `_clip_to_end` for the
   one of #758 while a solve with jump times is traced: only accepted steps move onto
   t1. Solves without jump times keep Diffrax's own: on an A100, the swap makes small
   solves up to 7% slower.
   To remove: once Diffrax releases #758, the strict xfail
   `test_diffrax_keeps_rejected_steps` passes and fails the test suite. Raise the
   `diffrax` lower bound, then remove `keep_rejected_steps`, its helpers and its use.

Once 1 and 2 are removed, use `dx.ClipStepSizeController` in `diffrax_integrator.py`.
Once all three are removed, delete this module and its tests.

This module reads private parts of Diffrax. If a release changes them, a test fails as
a reminder: `test_private_search_exists` for `_find_idx_with_hint` (the controller then
falls back to Diffrax's own search), `test_private_clip_state` for the state of
`ClipStepSizeController` (`step_info`, `jump_info`, `reject_info`). If `_clip_to_end`
moves or changes its parameters, the solves keep Diffrax's own, and the cases with a
jump 2 ULP before t1 of `test_discontinuity_at_last_save_time` fail.

The swaps change module attributes of Diffrax while a solve is traced: tracing solves
from several threads at once is not supported.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Generator

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

# private, as `_find_idx_with_hint`: Diffrax has no public way to change `_clip_to_end`
try:
    import diffrax._integrate as dx_integrate
except ImportError:  # moved in a newer Diffrax: keep its own `_clip_to_end`
    dx_integrate = None


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
    # Why: a step clipped to end just before a jump time can be tiny, but the
    # derivative of its length, d(jump) - d(t0), is of order 1. `PIDController` then
    # carries this derivative into every following step.
    # What: after a step that landed on a jump time or a `step_ts` time, give the
    # proposed `next_t1` the derivative of `next_t0`, keeping its value. The following
    # steps then move with the time just landed on.
    # This gives the same values as diffrax#773, bit for bit, including its two
    # exceptions: a `next_t1` clipped to the next jump time keeps the clip's
    # derivative, and so does one clipped to the next `step_ts` time, unless the step
    # landed on a jump.
    # It assumes, as for every controller dynamiqs builds, that the inner controller
    # never reports `made_jump`, and that rejected steps are not stored.
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


# the parameters of `_clip_to_end` in Diffrax 0.7.2; #758 removes `tprev`
_CLIP_TO_END_PARAMETERS = ('tprev', 'tnext', 't1', 't1_clip_floor', 'keep_step')
_diffrax_clip_to_end = getattr(dx_integrate, '_clip_to_end', None)


def does_diffrax_move_rejected_steps() -> bool:
    """Whether Diffrax's private `_clip_to_end` has the parameters of Diffrax 0.7.2.
    That version moves rejected steps onto t1 too.
    """
    if not callable(_diffrax_clip_to_end):
        return False
    parameters = tuple(inspect.signature(_diffrax_clip_to_end).parameters)
    return parameters == _CLIP_TO_END_PARAMETERS


def _clip_to_end_accepted_only(
    tprev: Array, tnext: Array, t1: Array, t1_clip_floor: Array, keep_step: Array
) -> Array:
    # the `_clip_to_end` of diffrax#758, with the parameters of Diffrax 0.7.2
    del tprev
    return jnp.where(keep_step & (tnext > t1_clip_floor), t1, tnext)


@contextlib.contextmanager
def keep_rejected_steps(enabled: bool) -> Generator[None]:
    """If `enabled`, use the `_clip_to_end` of diffrax#758 in the solves traced in
    this context.
    """
    if not enabled or not does_diffrax_move_rejected_steps():
        yield
        return
    dx_integrate._clip_to_end = _clip_to_end_accepted_only  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        dx_integrate._clip_to_end = _diffrax_clip_to_end  # ty: ignore[invalid-assignment]
