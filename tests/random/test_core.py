import jax
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    'random_qarray', [dq.random.ket, dq.random.dm, dq.random.operator]
)
def test_composite_dims(random_qarray):
    x = random_qarray(jax.random.PRNGKey(42), (2, 3))
    assert x.dims == (2, 3)
