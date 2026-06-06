import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
def test_against_mesolve_deexcitation(atol=4e-2):
    ntrajs = 1000
    gamma = 0.5
    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [jnp.sqrt(gamma) * dq.sigmam()]
    psi0 = dq.excited()
    tsave = jnp.linspace(0.0, 1.0, 11)
    keys = jax.random.split(jax.random.key(2081), num=ntrajs)
    exp_ops = [dq.excited_dm(), dq.ground_dm()]
    method = dq.method.EulerMaruyama(dt=1e-2)

    dsseresult = dq.dssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(
        H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=None
    )

    assert jnp.allclose(dsseresult.mean_expects(), meresult.expects, atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_no_back_action_protected_subspace(atol=1e-3):
    # The measurement operator is the identity on the odd-parity subspace spanned by
    # |01> and |10>, so the diffusive stochastic terms cancel trajectory by trajectory.
    ntrajs = 8
    omega = 1.3

    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * (dq.tensor(sx, sx) + dq.tensor(sy, sy))
    jump_ops = [-dq.tensor(sz, sz)]
    psi01 = dq.fock((2, 2), (0, 1))
    psi10 = dq.fock((2, 2), (1, 0))
    tsave = jnp.linspace(0.0, 1.2, 13)
    keys = jax.random.split(jax.random.key(2079), num=ntrajs)
    method = dq.method.Rouchon1(dt=1e-2)

    result = dq.dssesolve(H, jump_ops, psi01, tsave, keys=keys, method=method)

    exact = (
        jnp.cos(omega * tsave)[:, None, None] * psi01.to_jax()
        - 1j * jnp.sin(omega * tsave)[:, None, None] * psi10.to_jax()
    )
    exact = dq.asqarray(exact, dims=(2, 2))

    infidelity = 1.0 - dq.overlap(exact, result.states)
    assert jnp.allclose(infidelity, 0.0, atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_measurement_record_statistics():
    # For an eigenstate of a Hermitian measurement operator, the state is unchanged and
    # the saved averaged current satisfies I = <L + L†> + ΔW / Δt. Thus its mean and
    # variance are known analytically over each saved interval.
    ntrajs = 800
    gamma = 0.7

    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [jnp.sqrt(gamma) * dq.sigmaz()]
    psi0 = dq.excited()
    tsave = jnp.linspace(0.0, 1.0, 11)
    delta_t = tsave[1] - tsave[0]
    keys = jax.random.split(jax.random.key(2080), num=ntrajs)
    method = dq.method.EulerMaruyama(dt=1e-2)

    result = dq.dssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=[dq.excited_dm()], method=method
    )

    current = result.measurements[:, 0, :]
    expected_mean = 2 * jnp.sqrt(gamma)
    expected_variance = 1.0 / delta_t
    mean_standard_error = jnp.sqrt(expected_variance / ntrajs)
    variance_standard_error = expected_variance * jnp.sqrt(2.0 / (ntrajs - 1))

    assert jnp.allclose(result.expects[:, 0, :].real, 1.0, atol=1e-6)
    assert jnp.allclose(
        current.mean(axis=0), expected_mean, atol=3.0 * mean_standard_error
    )
    assert jnp.allclose(
        current.var(axis=0), expected_variance, atol=3.0 * variance_standard_error
    )
