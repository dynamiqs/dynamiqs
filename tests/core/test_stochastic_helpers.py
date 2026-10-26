import jax
import jax.numpy as jnp
import pytest

from dynamiqs import clicktimes_sse_to_sme, measurements_sse_to_sme

from ..order import TEST_SHORT


@pytest.mark.run(order=TEST_SHORT)
def test_clicktimes_sse_to_sme():
    # setup inputs
    key = jax.random.PRNGKey(42)
    tsave = jnp.linspace(0.0, 1.0, 11)
    etas = jnp.array([1.0, 0.0, 0.5])
    thetas = jnp.array([0.0, 1.0, 50.0])

    # clicktimes of shape (2 trajectories, 3 channels, 4 max clicks)
    clicktimes_sse = jnp.array(
        [
            [
                [0.1, 0.2, jnp.nan, jnp.nan],
                [0.5, 0.6, 0.7, jnp.nan],
                [0.8, 0.9, jnp.nan, jnp.nan],
            ],
            [
                [0.3, 0.4, jnp.nan, jnp.nan],
                [0.1, 0.2, jnp.nan, jnp.nan],
                [0.5, 0.55, jnp.nan, jnp.nan],
            ],
        ]
    )

    # compute results
    clicktimes_sme = clicktimes_sse_to_sme(clicktimes_sse, tsave, thetas, etas, key)

    # check outputs
    # eta = 0.0 channel should be removed
    assert clicktimes_sme.shape == (2, 2, 4)
    # eta = 1.0 channel should be identical
    assert jnp.allclose(
        clicktimes_sme[:, 0, :], clicktimes_sse[:, 0, :], equal_nan=True
    )
    # eta = 0.5 channel should be different
    assert not jnp.allclose(
        clicktimes_sme[:, 1, :], clicktimes_sse[:, 2, :], equal_nan=True
    )

    # check that click times are within tsave range
    valid_mask = ~jnp.isnan(clicktimes_sme[:, 1, :])
    assert jnp.all(clicktimes_sme[:, 1, :][valid_mask] >= tsave[0])
    assert jnp.all(clicktimes_sme[:, 1, :][valid_mask] <= tsave[-1])


@pytest.mark.run(order=TEST_SHORT)
def test_measurements_sse_to_sme():
    # setup inputs
    key = jax.random.PRNGKey(42)
    tsave = jnp.linspace(0.0, 1.0, 11)
    etas = jnp.array([1.0, 0.0, 0.5])

    # measurements of shape (2 trajectories, 3 channels, 10 time steps)
    measurements_sse = jnp.ones((2, 3, 10))

    # compute results
    measurements_sme = measurements_sse_to_sme(measurements_sse, tsave, etas, key)

    # check outputs
    # eta = 0.0 channel should be removed
    assert measurements_sme.shape == (2, 2, 10)
    # eta = 1.0 channel should be identical
    assert jnp.allclose(measurements_sme[:, 0, :], measurements_sse[:, 0, :])
    # eta = 0.5 channel should be different
    assert not jnp.allclose(measurements_sme[:, 1, :], measurements_sse[:, 2, :])
    # check batching: noise added to different trajectories should be different
    assert not jnp.allclose(measurements_sme[0, 1, :], measurements_sme[1, 1, :])
