import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize('cartesian_batching', [True, False])
def test_batching(cartesian_batching):
    n = 8
    ntsave = 11
    nH = 3
    npsi0 = 4 if cartesian_batching else nH
    nEs = 5

    options = dq.options.Options(cartesian_batching=cartesian_batching)

    # create random objects
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    H = dq.rand_herm(k1, (nH, n, n))
    exp_ops = dq.rand_complex(k2, (nEs, n, n))
    psi0 = dq.rand_ket(k3, (npsi0, n, 1))
    tsave = jnp.linspace(0, 0.01, ntsave)

    # no batching
    result = dq.sesolve(H[0], psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (ntsave, n, 1)
    assert result.expects.shape == (nEs, ntsave)

    # H batched
    result = dq.sesolve(H, psi0[0], tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (nH, ntsave, n, 1)
    assert result.expects.shape == (nH, nEs, ntsave)

    # psi0 batched
    result = dq.sesolve(H[0], psi0, tsave, exp_ops=exp_ops, options=options)
    assert result.states.shape == (npsi0, ntsave, n, 1)
    assert result.expects.shape == (npsi0, nEs, ntsave)

    # H and psi0 batched
    result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops, options=options)
    if cartesian_batching:
        assert result.states.shape == (nH, npsi0, ntsave, n, 1)
        assert result.expects.shape == (nH, npsi0, nEs, ntsave)
    else:
        assert result.states.shape == (nH, ntsave, n, 1)
        assert result.expects.shape == (nH, nEs, ntsave)


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
    result = dq.sesolve(H0 + H_cte, psi0, times)
    assert result.states.shape == (2, 11, 4, 1)

    # == pwc time array
    values = jnp.arange(3 * 10).reshape(3, 10)
    H_pwc = dq.pwc(times, values, H0)

    result = dq.sesolve(H_pwc, psi0, times)
    assert result.states.shape == (3, 11, 4, 1)
    result = dq.sesolve(H0 + H_pwc, psi0, times)
    assert result.states.shape == (3, 11, 4, 1)

    # == modulated time array
    deltas = jnp.linspace(0.0, 1.0, 4)
    H_mod = dq.modulated(lambda t, delta: jnp.cos(t * delta), H0, args=(deltas,))

    result = dq.sesolve(H_mod, psi0, times)
    assert result.states.shape == (4, 11, 4, 1)
    result = dq.sesolve(H0 + H_mod, psi0, times)
    assert result.states.shape == (4, 11, 4, 1)

    # == callable time array
    omegas = jnp.linspace(0.0, 1.0, 5)
    H_cal = dq.timecallable(
        lambda t, omega: jnp.cos(t * omega[..., None, None]) * H0, args=(omegas,)
    )

    result = dq.sesolve(H_cal, psi0, times)
    assert result.states.shape == (5, 11, 4, 1)
    result = dq.sesolve(H0 + H_cal, psi0, times)
    assert result.states.shape == (5, 11, 4, 1)
