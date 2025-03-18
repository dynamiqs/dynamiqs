import jax.numpy as jnp
import jax.random
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('smart_sampling', [True, False])
def test_against_mesolve_oscillator(smart_sampling, atol=5e-2):
    # parameters
    ntrajs = 80
    dim = 10

    # solver inputs
    a = dq.destroy(dim)
    H = 0.1 * a.dag() @ a + 0.4 * (a + a.dag())
    jump_ops = [a, 0.3 * a.dag()]
    psi0 = dq.basis(dim, 0)
    tsave = jnp.linspace(0.0, 2.0, 11)
    keys = jax.random.split(jax.random.key(31), num=ntrajs)
    exp_ops = [a.dag() @ a]
    js_options = dq.Options(smart_sampling=smart_sampling)
    me_options = dq.Options(progress_meter=None)

    # solve with jssesolve and mesolve
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    solver = dq.method.Event(root_finder=root_finder)
    jsseresult = dq.jssesolve(
        H,
        jump_ops,
        psi0,
        tsave,
        keys,
        exp_ops=exp_ops,
        options=js_options,
        method=solver,
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=me_options)

    # compare results on average
    if smart_sampling:
        mean_jsse_expects, mean_jsse_states = _average_smart_sampling(jsseresult)
    else:
        mean_jsse_expects = jnp.mean(jsseresult.expects, axis=0)
        mean_jsse_states = jsseresult.states.todm().sum(axis=0) / ntrajs
    assert jnp.allclose(meresult.expects, mean_jsse_expects, atol=atol)
    assert jnp.allclose(meresult.states.to_jax(), mean_jsse_states.to_jax(), atol=atol)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('smart_sampling', [True, False])
def test_against_mesolve_qubit(smart_sampling, atol=5e-2):
    # parameters
    ntrajs = 40
    omega = 2.0 * jnp.pi
    amp = 0.1 * 2.0 * jnp.pi

    # solver inputs
    def H_func(t):
        return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

    H = dq.timecallable(H_func)
    jump_ops = [0.4 * dq.sigmam()]
    psi0 = [dq.ground(), dq.excited()]
    tsave = jnp.linspace(0, 1.0, 41)
    keys = jax.random.split(jax.random.key(42), num=ntrajs)
    exp_ops = [dq.excited().todm(), dq.ground().todm()]
    js_options = dq.Options(smart_sampling=smart_sampling)
    me_options = dq.Options(progress_meter=None)

    # solve with jssesolve and mesolve
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, options=js_options
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=me_options)

    # compare results on average
    if smart_sampling:
        mean_jsse_expects, mean_jsse_states = _average_smart_sampling(jsseresult)
    else:
        mean_jsse_expects = jnp.mean(jsseresult.expects, axis=1)
        mean_jsse_states = jsseresult.states.todm().sum(axis=1) / ntrajs
    assert jnp.allclose(meresult.expects, mean_jsse_expects, atol=atol)
    assert jnp.allclose(meresult.states.to_jax(), mean_jsse_states.to_jax(), atol=atol)


def _average_smart_sampling(jsseresult):
    noclick_prob = jsseresult.final_state_norm[..., 0] ** 2
    expect_len = len(jsseresult.expects.shape)
    noclick_prob_expect = noclick_prob[(...,) + (None,) * (expect_len - 2)]
    noclick_prob_state = noclick_prob[(...,) + (None,) * (expect_len - 1)]
    mean_jsse_expects_click = jnp.mean(jsseresult.expects[..., 1:, :, :], axis=-3)
    mean_jsse_expects = (
        noclick_prob_expect * jsseresult.expects[..., 0, :, :]
        + (1 - noclick_prob_expect) * mean_jsse_expects_click
    )
    mean_jsse_states_click = jsseresult.states[..., 1:, :, :, :].todm().mean(axis=-4)
    mean_jsse_states = dq.unit(
        noclick_prob_state * jsseresult.states[..., 0, :, :, :].todm()
        + (1 - noclick_prob_state) * mean_jsse_states_click
    )
    return mean_jsse_expects, mean_jsse_states
