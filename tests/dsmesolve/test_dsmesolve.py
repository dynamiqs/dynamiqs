import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
def test_against_mesolve_deexcitation(atol=3e-2):
    ntrajs = 1_000
    gamma = 0.5

    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [jnp.sqrt(gamma) * dq.sigmam()]
    etas = [1.0]
    rho0 = dq.excited_dm()
    tsave = jnp.linspace(0.0, 1.0, 11)
    keys = jax.random.split(jax.random.key(4081), num=ntrajs)
    exp_ops = [dq.excited_dm(), dq.ground_dm()]
    method = dq.method.EulerMaruyama(dt=1e-2)

    dsmeresult = dq.dsmesolve(
        H, jump_ops, etas, rho0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(
        H, jump_ops, rho0, tsave, exp_ops=exp_ops, progress_meter=None
    )

    assert jnp.allclose(dsmeresult.mean_expects(), meresult.expects, atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_no_back_action_protected_subspace(atol=1e-2):
    # The measured operator acts as the identity on the odd-parity subspace spanned by
    # |01> and |10>, so the diffusive stochastic terms cancel and preserve the
    # deterministic Hamiltonian trajectory inside that subspace.
    ntrajs = 8
    omega = 1.3

    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * (dq.tensor(sx, sx) + dq.tensor(sy, sy))
    jump_ops = [-dq.tensor(sz, sz)]
    etas = [1.0]
    psi01 = dq.fock((2, 2), (0, 1))
    psi10 = dq.fock((2, 2), (1, 0))
    rho0 = psi01.todm()
    tsave = jnp.linspace(0.0, 1.2, 13)
    keys = jax.random.split(jax.random.key(4080), num=ntrajs)
    method = dq.method.Rouchon1(dt=2e-3)

    result = dq.dsmesolve(H, jump_ops, etas, rho0, tsave, keys=keys, method=method)

    exact = (
        jnp.cos(omega * tsave)[:, None, None] * psi01.to_jax()
        - 1j * jnp.sin(omega * tsave)[:, None, None] * psi10.to_jax()
    )
    exact = dq.asqarray(exact, dims=(2, 2)).todm()

    assert jnp.allclose(result.states.to_jax(), exact.to_jax(), atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_measurement_record_statistics():
    # For an eigenstate of a Hermitian measurement operator, the state is unchanged and
    # the saved averaged current satisfies I = sqrt(eta) <L + L†> + ΔW / Δt.
    ntrajs = 800
    gamma = 0.7
    eta = 0.25

    H = dq.zeros_like(dq.sigmaz())
    jump_ops = [jnp.sqrt(gamma) * dq.sigmaz()]
    etas = [eta]
    rho0 = dq.excited_dm()
    tsave = jnp.linspace(0.0, 1.0, 11)
    delta_t = tsave[1] - tsave[0]
    keys = jax.random.split(jax.random.key(4082), num=ntrajs)
    method = dq.method.EulerMaruyama(dt=1e-2)

    result = dq.dsmesolve(
        H,
        jump_ops,
        etas,
        rho0,
        tsave,
        keys=keys,
        exp_ops=[dq.excited_dm()],
        method=method,
    )

    current = result.measurements[:, 0, :]
    expected_mean = 2.0 * jnp.sqrt(eta) * jnp.sqrt(gamma)
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
