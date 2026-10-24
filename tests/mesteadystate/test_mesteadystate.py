import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT

n = 4
a = dq.destroy(n)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    'H',
    [dq.modulated(jnp.cos, a + a.dag()), dq.constant(a.dag() @ a).clip(0.0, 1.0)],
    ids=['modulated', 'clipped_constant'],
)
def test_time_dependent_H(H):
    with pytest.raises(TypeError, match='Argument `H` must be time-independent'):
        dq.mesteadystate(H, [a])
