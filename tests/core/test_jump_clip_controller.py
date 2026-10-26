import diffrax as dx
import diffrax._step_size_controller.clip as dx_clip
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core.jump_clip_controller import (
    JumpClipController,
    _find_idx_by_counting,
    private_search_available,
)

from ..order import TEST_INSTANT, TEST_LONG, TEST_SHORT


@pytest.mark.run(order=TEST_INSTANT)
def test_find_idx_matches_diffrax():
    # the loop-free search returns Diffrax's index, wherever t falls (before, between,
    # on and after the jump times) and whatever the hint
    ts = jnp.array([0.1, 0.4, 0.4, 0.7, 1.5])
    for t in [-1.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for hint in range(len(ts) + 1):
            expected = dx_clip._find_idx_with_hint(t, ts, jnp.array(hint))
            assert int(_find_idx_by_counting(t, ts, jnp.array(hint))) == int(expected)
    assert _find_idx_by_counting(0.3, None, 0) == np.int64(0)


@pytest.mark.run(order=TEST_INSTANT)
def test_private_search_exists():
    # JumpClipController replaces this private function of Diffrax; if a release moved
    # or renamed it, the controller falls back to Diffrax's search (with while loops):
    # find its new home, or drop the controller if the new search has no while loop
    assert private_search_available()


@pytest.mark.run(order=TEST_INSTANT)
def test_private_clip_state():
    # the private state `_drop_step_tangent` reads: if this fails, update it
    step_ts = jnp.array([0.5, 0.8])
    controller = dx.ClipStepSizeController(
        dx.PIDController(rtol=1e-3, atol=1e-3),
        step_ts=step_ts,
        jump_ts=jnp.array([0.9]),
    )
    term = dx.ODETerm(lambda t, y, args: -y)  # noqa: ARG005
    y0 = jnp.array(1.0)
    _, state = controller.init(term, 0.0, 1.0, y0, 0.5, None, term.vf, 5)
    keep_step, next_t0, *_, state, _ = controller.adapt_step_size(
        0.0, 0.5, y0, y0, None, jnp.array(0.0), 5, state
    )
    assert keep_step
    assert next_t0 == 0.5
    index, ts = state.step_info
    assert index == 1
    assert jnp.array_equal(ts, step_ts)
    assert state.jump_info is not None
    assert state.reject_info is None


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.xfail(
    reason="diffrax#776: Diffrax's search of the next `step_ts` or `jump_ts` time "
    'compiles to while loops; an XPASS means the installed Diffrax has none: see the '
    'docstring of `jump_clip_controller.py` for what to remove',
    raises=AssertionError,
    strict=True,
)
def test_diffrax_search_is_loop_free():
    # the lowered program, not the jaxpr: a `scan` or `fori_loop` is no `while` in the
    # jaxpr, but still an XLA while loop
    if not private_search_available():
        pytest.skip('Diffrax moved its search: see `test_private_search_exists`')
    ts = jnp.linspace(0.0, 1.0, 10)
    search = jax.jit(lambda t: dx_clip._find_idx_with_hint(t, ts, 0))
    assert 'stablehlo.while' not in search.lower(0.5).as_text()


def rotation(t, y, jump_t):
    # a rotation at speed 1.6 up to 1.5, -1.1 up to `jump_t`, then 0
    h = jnp.where(t < 1.5, 1.6, jnp.where(t < jump_t, -1.1, 0.0))
    return h * jnp.stack([-y[1], y[0]])


def solve_rotation_with_diffrax(jump_t):
    # Diffrax alone, up to t1 = 3 with `jump_t` declared, in single precision
    controller = dx.ClipStepSizeController(
        dx.PIDController(rtol=1e-9, atol=1e-9), jump_ts=jnp.stack([1.5, jump_t])
    )
    return dx.diffeqsolve(
        dx.ODETerm(rotation),
        dx.Tsit5(),
        0.0,
        3.0,
        None,
        jnp.array([1.0, 0.0]),
        args=jump_t,
        saveat=dx.SaveAt(ts=jnp.linspace(0.0, 3.0, 11)),
        stepsize_controller=controller,
        throw=False,
    )


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.xfail(
    reason='diffrax#791: with a jump at t1, the saved state at t1 stays `inf`; an '
    'XPASS means the installed Diffrax saves it: remove the block "end the solve '
    'just before a jump at t1" of `DiffraxIntegrator.diffeqsolve`',
    raises=AssertionError,
    strict=True,
)
def test_diffrax_saves_at_jump_at_t1():
    solution = solve_rotation_with_diffrax(jnp.float32(3.0))
    assert jnp.all(jnp.isfinite(solution.ys[-1]))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.xfail(
    reason='diffrax#756: a jump in the last 100 ULP of the solve makes Diffrax reject '
    'the same step until `max_steps`; an XPASS means the installed Diffrax has '
    '#758: see the docstring of `jump_clip_controller.py` for what to remove',
    raises=AssertionError,
    strict=True,
)
def test_diffrax_keeps_rejected_steps():
    # a jump 2 to 9 ULP before t1
    jump_t = jnp.nextafter(jnp.float32(3.0), -jnp.inf)
    for _ in range(8):
        jump_t = jnp.nextafter(jump_t, -jnp.inf)
        solution = solve_rotation_with_diffrax(jump_t)
        assert solution.result == dx.RESULTS.successful


diffrax_tangent_blowup = pytest.mark.xfail(
    reason='diffrax#772: the derivative with respect to close traced jump times blows '
    'up; an XPASS means the installed Diffrax fixed it: see the docstring of '
    '`jump_clip_controller.py` for what to remove',
    raises=AssertionError,
    strict=True,
)


@pytest.fixture
def _double_precision():
    # the `step_ts` half of `_drop_step_tangent` only shows below single precision
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')
    yield
    dq.set_precision('double' if prev_x64 else 'single')


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.usefixtures('_double_precision')
@pytest.mark.parametrize(
    ('controller', 'mode', 'with_step_ts'),
    [
        # one case is enough to tell when Diffrax is fixed
        pytest.param(
            dx.ClipStepSizeController, 'forward', False, marks=diffrax_tangent_blowup
        ),
        # one case per half of `_drop_step_tangent`
        (JumpClipController, 'forward', False),
        (JumpClipController, 'reverse', True),
    ],
    ids=['diffrax', 'dynamiqs-forward', 'dynamiqs-reverse'],
)
def test_derivative_wrt_close_jump_times(controller, mode, with_step_ts):
    # y' = -20 y + 15 on 7 pulses 0.04 apart, whose 14 edges all move with `shift`;
    # from y(2) = Σ_k 0.75 (exp(-20 (2 - end_k)) - exp(-20 (2 - start_k))),
    # dy(2)/dshift = Σ_k 15 (exp(-20 (2 - end_k)) - exp(-20 (2 - start_k)))
    def edges(shift):
        length = (1.78 - 6 * 0.04) / 7
        starts = 0.1 + jnp.arange(7.0) * (length + 0.04) + shift
        return starts, starts + length

    def y_end(shift):
        starts, ends = edges(shift)

        def vector_field(t, y, args):
            pulses = (t >= starts) & (t < ends)
            return -20.0 * y + 15.0 * jnp.sum(jnp.where(pulses, 1.0, 0.0))

        stepsize_controller = controller(
            dx.PIDController(rtol=1e-8, atol=1e-8),
            jump_ts=jnp.sort(jnp.concatenate([starts, ends])),
            step_ts=jnp.linspace(0.0, 2.0, 101) if with_step_ts else None,
        )
        adjoint = (
            dx.ForwardMode() if mode == 'forward' else dx.RecursiveCheckpointAdjoint()
        )
        sol = dx.diffeqsolve(
            dx.ODETerm(vector_field),
            dx.Tsit5(),
            0.0,
            2.0,
            None,
            0.0,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
        )
        return sol.ys[0]

    shift = 0.013
    starts, ends = edges(shift)
    exact = jnp.sum(
        15.0 * (jnp.exp(-20.0 * (2.0 - ends)) - jnp.exp(-20.0 * (2.0 - starts)))
    )
    if mode == 'forward':
        _, derivative = jax.jvp(y_end, (shift,), (1.0,))
    else:
        derivative = jax.grad(y_end)(shift)
    assert jnp.allclose(derivative, exact, rtol=1e-6)
