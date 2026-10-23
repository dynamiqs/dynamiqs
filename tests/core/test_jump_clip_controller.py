import diffrax._step_size_controller.clip as dx_clip
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dynamiqs.integrators.core.jump_clip_controller import (
    _find_idx_by_counting,
    private_search_available,
)

from ..order import TEST_INSTANT


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
