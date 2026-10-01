import diffrax._step_size_controller.clip as dx_clip
import jax.numpy as jnp
import numpy as np
import pytest

from dynamiqs.integrators.core.jump_clip_controller import _find_idx_by_counting

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
