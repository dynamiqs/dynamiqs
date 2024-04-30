import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,), (5, 6)])
def test_batching(cartesian_batching, nH, npsi0):
    n = 8
    ntsave = 11
    npsi0 = npsi0 if cartesian_batching else nH
    nE = 7

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    H = dq.rand_herm(k1, (*nH, n, n))
    exp_ops = dq.rand_complex(k2, (nE, n, n))
    psi0 = dq.rand_ket(k3, (*npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, ntsave)

    result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (*nH, *npsi0, ntsave, n, 1)
        assert result.expects.shape == (*nH, *npsi0, nE, ntsave)
    else:
        assert result.states.shape == (*nH, ntsave, n, 1)
        assert result.expects.shape == (*nH, nE, ntsave)


def test_timearray_batching():
    # generic arrays
    a = dq.destroy(4)
    H0 = a + dq.dag(a)
    psi0 = dq.basis(4, 0)
    times = jnp.linspace(0.0, 1.0, 11)

    # == constant time array
    H_cte = jnp.stack([H0, 2 * H0])

    result = dq.sesolve(H_cte, psi0, times)
    assert result.states.shape == (2, 11, 4, 1)
    # todo: fix
    # result = dq.sesolve(H0 + H_cte, psi0, times)
    # assert result.states.shape == (2, 11, 4, 1)

    # == pwc time array
    values = jnp.arange(3 * 10).reshape(3, 10)
    H_pwc = dq.pwc(times, values, H0)

    result = dq.sesolve(H_pwc, psi0, times)
    assert result.states.shape == (3, 11, 4, 1)
    # todo: fix
    # result = dq.sesolve(H0 + H_pwc, psi0, times)
    # assert result.states.shape == (3, 11, 4, 1)

    # == modulated time array
    deltas = jnp.linspace(0.0, 1.0, 4)
    H_mod = dq.modulated(lambda t: jnp.cos(t * deltas), H0)

    # todo :fix
    result = dq.sesolve(H_mod, psi0, times)
    assert result.states.shape == (4, 11, 4, 1)
    # todo: fix
    # result = dq.sesolve(H0 + H_mod, psi0, times)
    # assert result.states.shape == (4, 11, 4, 1)

    # == callable time array
    omegas = jnp.linspace(0.0, 1.0, 5)
    H_cal = dq.timecallable(lambda t: jnp.cos(t * omegas[..., None, None]) * H0)

    result = dq.sesolve(H_cal, psi0, times)
    assert result.states.shape == (5, 11, 4, 1)
    # todo: fix
    # result = dq.sesolve(H0 + H_cal, psi0, times)
    # assert result.states.shape == (5, 11, 4, 1)
