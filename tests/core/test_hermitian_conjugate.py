import jax.numpy as jnp
import pytest

from dynamiqs import QArray, dense, hc, sigmay
from tests.order import TEST_INSTANT


def assert_equal(x, y):
    if isinstance(x, QArray):
        x = x.to_jax()
    if isinstance(y, QArray):
        y = y.to_jax()
    assert jnp.array_equal(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_hc_qarray():
    x = sigmay(layout=dense)
    assert_equal(x + hc, x + x.dag())
