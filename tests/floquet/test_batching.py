import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(('nH'), [(), (3,), (3, 4)])
@pytest.mark.parametrize('H_type', ['constant', 'modulated', 'timecallable'])
def test_batching(nH, H_type):
    n = 2
    T = 1.0
    tsave = jnp.linspace(0.0, T, 5)
    key = jax.random.PRNGKey(84)
    key_1, key_2, key_3 = jax.random.split(key, 3)
    if H_type == 'constant':
        H = dq.random.herm(key_1, (*nH, n, n))
    elif H_type == 'modulated':
        _H = dq.random.herm(key_1, (n, n))
        f_pref = dq.random.real(key_2, (*nH,))
        omega_d = 2.0 * jnp.pi / T
        H = dq.modulated(lambda t: f_pref * jnp.cos(omega_d * t), _H)
    else:  # timecallable
        _H = dq.random.herm(key_1, (*nH, n, n))
        omega_d = 2.0 * jnp.pi / T
        H = dq.timecallable(lambda t: jnp.cos(omega_d * t)[..., None, None] * _H)

    result = dq.floquet(H, T, tsave=tsave)
    assert result.modes.shape == (*nH, tsave.shape[-1], n, n, 1)
    assert result.quasienergies.shape == (*nH, n)
