import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
def test_no_back_action_protected_subspace(atol=1e-2):
    # The jump operator is the identity on the odd-parity subspace spanned by
    # |01> and |10>, so stochastic clicks must not perturb the deterministic
    # Hamiltonian trajectory inside that subspace.
    ntrajs = 8
    omega = 1.3

    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * (dq.tensor(sx, sx) + dq.tensor(sy, sy))
    jump_ops = [-dq.tensor(sz, sz)]
    thetas = [0.0]
    etas = [1.0]
    psi01 = dq.fock((2, 2), (0, 1))
    psi10 = dq.fock((2, 2), (1, 0))
    rho0 = psi01.todm()
    tsave = jnp.linspace(0.0, 1.2, 13)
    keys = jax.random.split(jax.random.key(3080), num=ntrajs)
    method = dq.method.EulerJump(dt=2e-3)

    result = dq.jsmesolve(
        H, jump_ops, thetas, etas, rho0, tsave, keys=keys, method=method, nmaxclick=16
    )

    exact = (
        jnp.cos(omega * tsave)[:, None, None] * psi01.to_jax()
        - 1j * jnp.sin(omega * tsave)[:, None, None] * psi10.to_jax()
    )
    exact = dq.asqarray(exact, dims=(2, 2)).todm()

    assert jnp.allclose(result.states.to_jax(), exact.to_jax(), atol=atol)
    assert jnp.any(result.nclicks > 0)


@pytest.mark.run(order=TEST_LONG)
def test_deexcitation_bernoulli_statistics(atol=5e-2):
    # For an ideal detector monitoring L=sqrt(gamma) sigma_-, each trajectory is either
    # still excited or has emitted one jump. With the fixed-step jump method, the exact
    # no-click probability at saved times is (1 - gamma * dt) ** nsteps.
    ntrajs = 800
    gamma = 0.8
    dt = 1e-2

    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [jnp.sqrt(gamma) * dq.sigmam()]
    thetas = [0.0]
    etas = [1.0]
    rho0 = dq.excited_dm()
    tsave = jnp.linspace(0.0, 2.0, 21)
    keys = jax.random.split(jax.random.key(3081), num=ntrajs)
    method = dq.method.EulerJump(dt=dt)

    result = dq.jsmesolve(
        H,
        jump_ops,
        thetas,
        etas,
        rho0,
        tsave,
        keys=keys,
        exp_ops=[dq.excited_dm()],
        method=method,
        nmaxclick=2,
    )

    excited_population = result.expects[:, 0, :].real
    expected_mean = (1.0 - gamma * dt) ** jnp.rint(tsave / dt)
    expected_variance = expected_mean * (1.0 - expected_mean)

    assert jnp.allclose(excited_population * (1.0 - excited_population), 0.0, atol=1e-6)
    assert jnp.allclose(excited_population.mean(axis=0), expected_mean, atol=atol)
    assert jnp.allclose(excited_population.var(axis=0), expected_variance, atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_dark_count_statistics():
    # With a zero jump operator and nonzero dark-count rate, clicks are independent of
    # the quantum state. The number of clicks over the integration is binomial over the
    # fixed time steps with probability theta * dt per step.
    ntrajs = 1_000
    theta = 0.4
    dt = 1e-2
    t1 = 1.0

    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [dq.zeros_like(dq.sigmaz())]
    thetas = [theta]
    etas = [1.0]
    rho0 = dq.excited_dm()
    tsave = jnp.linspace(0.0, t1, 11)
    keys = jax.random.split(jax.random.key(3082), num=ntrajs)
    method = dq.method.EulerJump(dt=dt)

    result = dq.jsmesolve(
        H,
        jump_ops,
        thetas,
        etas,
        rho0,
        tsave,
        keys=keys,
        exp_ops=[dq.excited_dm()],
        method=method,
        nmaxclick=8,
    )

    nsteps = round(t1 / dt)
    click_probability = theta * dt
    expected_mean = nsteps * click_probability
    expected_variance = nsteps * click_probability * (1.0 - click_probability)
    mean_standard_error = jnp.sqrt(expected_variance / ntrajs)
    variance_standard_error = expected_variance * jnp.sqrt(2.0 / (ntrajs - 1))
    nclicks = result.nclicks[:, 0]

    assert jnp.allclose(result.expects[:, 0, :].real, 1.0, atol=1e-6)
    assert jnp.allclose(nclicks.mean(), expected_mean, atol=3.0 * mean_standard_error)
    assert jnp.allclose(
        nclicks.var(), expected_variance, atol=3.0 * variance_standard_error
    )
