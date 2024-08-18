import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs import floquet, floquet_t, timecallable


@pytest.mark.parametrize('nH', [(3,), (3, 4)])
def test_batching_tc(nH):
    n = 6
    ntsave = 5
    T = 3.4
    tsave = jnp.linspace(0.0, 1.0, ntsave)
    key = jax.random.PRNGKey(84)
    _H = dq.random.herm(key, (*nH, n, n))
    omega_d = 2.0 * jnp.pi / T

    H = timecallable(lambda t: jnp.cos(omega_d * t) * _H)
    result_0 = dq.floquet(H, T, safe=True)
    assert result_0.floquet_modes.shape == (*nH, n, n)
    assert result_0.quasi_energies.shape == (*nH, n)

    result_t = dq.floquet_t(H, T, tsave=tsave, safe=True)
    assert result_t.floquet_modes.shape == (*nH, ntsave, n, n)
    assert result_t.quasi_energies.shape == (*nH, n)

@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('T', [1.0, [1.0]])
def test_batching_constant(nH, T):
    n = 2
    ntsave = 5
    tsave = jnp.linspace(0.0, 1.0, ntsave)
    key = jax.random.PRNGKey(84)
    H = dq.random.herm(key, (*nH, n, n))
    broadcast_shape = jnp.broadcast_shapes(nH, jnp.array(T).shape)
    result_0 = dq.floquet(H, T, safe=True)
    assert result_0.floquet_modes.shape == (*broadcast_shape, n, n)
    assert result_0.quasi_energies.shape == (*broadcast_shape, n)

    result_t = dq.floquet_t(H, T, tsave=tsave, safe=True)
    assert result_t.floquet_modes.shape == (*broadcast_shape, ntsave, n, n)
    assert result_t.quasi_energies.shape == (*broadcast_shape, n)


# H has fixed shape (3, 4, n, n) for the next test case, we test if the Hamiltonian is
# periodic with differing frequencies for one of the batch dimensions
# TODO this is failing with an obscure diffrax error...
def test_T_batching():
    n = 2
    nH = (3, 4)
    nT = (4,)
    ntsave = 5
    (kH, kT) = jax.random.split(jax.random.PRNGKey(168), 2)
    _H = dq.random.herm(kH, (*nH, n, n))
    Ts = jax.random.uniform(kT, nT, minval=0.5, maxval=1.5)
    omega_ds = 2.0 * jnp.pi / Ts

    def H_func(t):
        return jnp.einsum('j,ijkl->ijkl', jnp.cos(omega_ds * t), _H)

    H = timecallable(H_func)

    result_0 = floquet(H, Ts, safe=True)
    assert result_0.floquet_modes.shape == (*nH, n, n)
    assert result_0.quasi_energies.shape == (*nH, n)

    tsave = [jnp.linspace(0.0, T, ntsave) for T in Ts]
    result_t = floquet_t(H, Ts, tsave=tsave, safe=True)
    assert result_t.floquet_modes.shape == (*nH, ntsave, n, n)
    assert result_t.quasi_energies.shape == (*nH, ntsave, n)
