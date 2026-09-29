import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG

# `H` is the only batchable argument of sepropagator, so `cartesian_batching` is not a
# supported option and there is no flat batching to test here


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
def test_cartesian_batching(nH):
    n = 2
    ntsave = 11

    # run sepropagator
    H = dq.random.operator(jax.random.PRNGKey(42), n, batch=nH)
    tsave = jnp.linspace(0, 0.01, ntsave)
    result = dq.sepropagator(H, tsave)

    # check result shape
    assert result.propagators.shape == (*nH, ntsave, n, n)


@pytest.mark.run(order=TEST_LONG)
def test_timeqarray_batching():
    # generic qarrays
    a = dq.destroy(4)
    H0 = a + a.dag()
    times = jnp.linspace(0.0, 1.0, 11)

    # == constant timeqarray
    H_cte = dq.stack([H0, 2 * H0])

    result = dq.sepropagator(H_cte, times)
    assert result.propagators.shape == (2, 11, 4, 4)
    result = dq.sepropagator(H0 + H_cte, times)
    assert result.propagators.shape == (2, 11, 4, 4)

    # == pwc timeqarray
    values = jnp.arange(3 * 10).reshape(3, 10)
    H_pwc = dq.pwc(times, values, H0)

    result = dq.sepropagator(H_pwc, times)
    assert result.propagators.shape == (3, 11, 4, 4)
    result = dq.sepropagator(H0 + H_pwc, times)
    assert result.propagators.shape == (3, 11, 4, 4)

    # == modulated timeqarray
    deltas = jnp.linspace(0.0, 1.0, 4)
    H_mod = dq.modulated(lambda t: jnp.cos(t * deltas), H0)

    result = dq.sepropagator(H_mod, times)
    assert result.propagators.shape == (4, 11, 4, 4)
    result = dq.sepropagator(H0 + H_mod, times)
    assert result.propagators.shape == (4, 11, 4, 4)

    # == callable timeqarray
    omegas = jnp.linspace(0.0, 1.0, 5)
    H_cal = dq.timecallable(lambda t: jnp.cos(t * omegas[..., None, None]) * H0)

    result = dq.sepropagator(H_cal, times)
    assert result.propagators.shape == (5, 11, 4, 4)
    result = dq.sepropagator(H0 + H_cal, times)
    assert result.propagators.shape == (5, 11, 4, 4)
