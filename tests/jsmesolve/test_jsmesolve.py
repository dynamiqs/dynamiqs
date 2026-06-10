import jax.numpy as jnp
import jax.random
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
def test_against_mesolve_qubit(atol=2e-2):
    # parameters
    ntrajs = 1000
    omega = 2.0 * jnp.pi
    amp = 0.1 * 2.0 * jnp.pi

    # solver inputs
    def H_func(t):
        return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

    H = dq.timecallable(H_func)
    jump_ops = [0.4 * dq.sigmam()]
    thetas = jnp.array([0.2])  # dark count rate
    etas = jnp.array([0.6])  # partial measurement efficiency
    rho0 = [dq.ground_dm(), dq.excited_dm()]
    tsave = jnp.linspace(0, 1.0, 41)
    keys = jax.random.split(jax.random.key(42), num=ntrajs)
    exp_ops = [dq.excited().todm(), dq.ground().todm()]
    method = dq.method.EulerJump(dt=1e-3)

    # solve with jsmesolve and mesolve (the ensemble average over the measurement
    # outcomes recovers the Lindblad evolution, for any efficiency eta and dark
    # count rate theta)
    jsmeresult = dq.jsmesolve(
        H, jump_ops, thetas, etas, rho0, tsave, keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(
        H, jump_ops, rho0, tsave, exp_ops=exp_ops, progress_meter=None
    )

    # compare results on average
    assert jnp.allclose(meresult.expects, jsmeresult.mean_expects(), atol=atol)
    assert jnp.allclose(
        meresult.states.to_jax(), jsmeresult.mean_states().to_jax(), atol=atol
    )
