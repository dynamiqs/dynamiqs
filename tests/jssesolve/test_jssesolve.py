import jax.numpy as jnp
import jax.random
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('smart_sampling', [True, False])
def test_against_mesolve_oscillator(smart_sampling, atol=1e-2):
    # parameters
    ntrajs = 2000
    dim = 10

    # solver inputs
    a = dq.destroy(dim)
    H = 0.1 * a.dag() @ a + 0.4 * (a + a.dag())
    jump_ops = [a, 0.3 * a.dag()]
    psi0 = dq.basis(dim, 0)
    tsave = jnp.linspace(0.0, 2.0, 11)
    keys = jax.random.split(jax.random.key(31), num=ntrajs)
    exp_ops = [a.dag() @ a]
    me_options = dq.Options(progress_meter=None)

    # solve with jssesolve and mesolve
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    method = dq.method.Event(root_finder=root_finder, smart_sampling=smart_sampling)
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=me_options)

    # compare results on average
    assert jnp.allclose(meresult.expects, jsseresult.mean_expects(), atol=atol)
    assert jnp.allclose(
        meresult.states.to_jax(), jsseresult.mean_states().to_jax(), atol=atol
    )


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('smart_sampling', [True, False])
def test_against_mesolve_qubit(smart_sampling, atol=1e-2):
    # parameters
    ntrajs = 1000
    omega = 2.0 * jnp.pi
    amp = 0.1 * 2.0 * jnp.pi

    # solver inputs
    def H_func(t):
        return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

    H = dq.timecallable(H_func)
    jump_ops = [0.4 * dq.sigmam()]
    psi0 = [dq.ground(), dq.excited()]
    tsave = jnp.linspace(0, 1.0, 41)
    keys = jax.random.split(jax.random.key(31), num=ntrajs)
    exp_ops = [dq.excited().todm(), dq.ground().todm()]
    me_options = dq.Options(progress_meter=None)
    root_finder = optx.Newton(1e-3, 1e-3, jtu.Partial(optx.rms_norm))
    method = dq.method.Event(root_finder=root_finder, smart_sampling=smart_sampling)

    # solve with jssesolve and mesolve
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=me_options)

    # compare results on average
    assert jnp.allclose(meresult.expects, jsseresult.mean_expects(), atol=atol)
    assert jnp.allclose(
        meresult.states.to_jax(), jsseresult.mean_states().to_jax(), atol=atol
    )
