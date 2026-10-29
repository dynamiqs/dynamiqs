import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('H', 'jump_ops', 'error', 'match'),
    [
        (
            dq.modulated(jnp.cos, dq.sigmax()),
            [dq.sigmam()],
            TypeError,
            'must be time-independent',
        ),
        (
            dq.constant(dq.sigmax()).clip(0.0, 1.0),
            [dq.sigmam()],
            TypeError,
            'must be time-independent',
        ),
        (dq.sigmax(), [], ValueError, 'at least one jump operator'),
    ],
    ids=['time-dependent', 'clipped-constant', 'no-jump-ops'],
)
def test_invalid_args(H, jump_ops, error, match):
    with pytest.raises(error, match=match):
        dq.steadystate(H, jump_ops)
