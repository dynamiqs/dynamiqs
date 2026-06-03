import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from .order import TEST_LONG, TEST_SHORT


def _odd_parity_model():
    omega = 0.7
    H = 0.5 * omega * (
        dq.tensor(dq.sigmax(), dq.sigmax()) + dq.tensor(dq.sigmay(), dq.sigmay())
    )
    measured_op = -dq.tensor(dq.sigmaz(), dq.sigmaz())
    psi0 = dq.tensor(dq.basis(2, 0), dq.basis(2, 1))
    tsave = jnp.linspace(0.0, 0.2, 5)

    ket01 = dq.tensor(dq.basis(2, 0), dq.basis(2, 1)).to_jax()
    ket10 = dq.tensor(dq.basis(2, 1), dq.basis(2, 0)).to_jax()
    exact_states = jnp.stack(
        [jnp.cos(omega * t) * ket01 - 1j * jnp.sin(omega * t) * ket10 for t in tsave]
    )
    exact_dms = dq.asqarray(exact_states).todm().to_jax()
    return H, measured_op, psi0, tsave, exact_dms


def _run_stochastic_solver(solver_name, H, jump_op, psi0, tsave, keys, *, exp_ops=None):
    rho0 = psi0.todm()

    if solver_name == 'jssesolve':
        return dq.jssesolve(
            H,
            [jump_op],
            psi0,
            tsave,
            keys=keys,
            exp_ops=exp_ops,
            method=dq.method.EulerJump(dt=0.025),
        )
    if solver_name == 'dssesolve':
        return dq.dssesolve(
            H,
            [jump_op],
            psi0,
            tsave,
            keys=keys,
            exp_ops=exp_ops,
            method=dq.method.EulerMaruyama(dt=0.025),
        )
    if solver_name == 'jsmesolve':
        return dq.jsmesolve(
            H,
            [jump_op],
            jnp.zeros(1),
            jnp.ones(1),
            rho0,
            tsave,
            keys=keys,
            exp_ops=exp_ops,
            method=dq.method.EulerJump(dt=0.025),
        )
    if solver_name == 'dsmesolve':
        return dq.dsmesolve(
            H,
            [jump_op],
            jnp.ones(1),
            rho0,
            tsave,
            keys=keys,
            exp_ops=exp_ops,
            method=dq.method.EulerMaruyama(dt=0.025),
        )

    raise ValueError(f'Unknown solver: {solver_name}')


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    'solver_name', ['jssesolve', 'dssesolve', 'jsmesolve', 'dsmesolve']
)
def test_qnd_measurement_has_no_backaction_on_odd_parity_trajectories(solver_name):
    H, measured_op, psi0, tsave, exact_dms = _odd_parity_model()
    keys = jax.random.split(jax.random.key(2026), 4)

    result = _run_stochastic_solver(
        solver_name, H, measured_op, psi0, tsave, keys
    ).block_until_ready()

    trajectory_dms = result.states.todm().to_jax()
    assert jnp.allclose(trajectory_dms, exact_dms[None, ...], atol=2e-2)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    'solver_name', ['jssesolve', 'dssesolve', 'jsmesolve', 'dsmesolve']
)
def test_stochastic_ensembles_match_mesolve_first_and_second_moments(solver_name):
    ntrajs = 96
    gamma = 0.7
    drive = 0.3
    H = 0.5 * drive * dq.sigmax()
    jump_op = jnp.sqrt(gamma) * dq.sigmam()
    psi0 = dq.excited()
    tsave = jnp.linspace(0.0, 0.3, 7)
    exp_ops = [dq.excited_dm()]

    reference = dq.mesolve(
        H,
        [jump_op],
        psi0.todm(),
        tsave,
        exp_ops=exp_ops,
        progress_meter=False,
    ).block_until_ready()

    result = _run_stochastic_solver(
        solver_name,
        H,
        jump_op,
        psi0,
        tsave,
        jax.random.split(jax.random.key(1079), ntrajs),
        exp_ops=exp_ops,
    ).block_until_ready()
    repeat = _run_stochastic_solver(
        solver_name,
        H,
        jump_op,
        psi0,
        tsave,
        jax.random.split(jax.random.key(1080), ntrajs),
        exp_ops=exp_ops,
    ).block_until_ready()

    assert jnp.allclose(result.mean_expects(), reference.expects, atol=7.5e-2)

    second_moment = jnp.mean(jnp.real(result.expects) ** 2, axis=0)
    repeat_second_moment = jnp.mean(jnp.real(repeat.expects) ** 2, axis=0)
    assert jnp.any(second_moment[:, 1:] > 1e-3)
    assert jnp.allclose(
        second_moment[:, 1:], repeat_second_moment[:, 1:], atol=7.5e-2
    )
