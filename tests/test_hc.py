import jax.numpy as jnp

from dynamiqs import QArray, dense, hc, sigmay


def assert_equal(x, y):
    if isinstance(x, QArray):
        x = x.to_jax()
    if isinstance(y, QArray):
        y = y.to_jax()
    assert jnp.array_equal(x, y)


def test_hc():
    x = sigmay(layout=dense)
    assert_equal(x + hc, x + x.dag())
