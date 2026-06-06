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

    # solve with jssesolve and mesolve
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    method = dq.method.Event(root_finder=root_finder, smart_sampling=smart_sampling)
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(
        H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=None
    )

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
    keys = jax.random.split(jax.random.key(42), num=ntrajs)
    exp_ops = [dq.excited().todm(), dq.ground().todm()]
    root_finder = optx.Newton(1e-3, 1e-3, jtu.Partial(optx.rms_norm))
    method = dq.method.Event(root_finder=root_finder, smart_sampling=smart_sampling)

    # solve with jssesolve and mesolve
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, method=method
    )
    meresult = dq.mesolve(
        H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=None
    )

    # compare results on average
    assert jnp.allclose(meresult.expects, jsseresult.mean_expects(), atol=atol)
    assert jnp.allclose(
        meresult.states.to_jax(), jsseresult.mean_states().to_jax(), atol=atol
    )


@pytest.mark.run(order=TEST_LONG)
def test_no_back_action_protected_subspace(atol=1e-5):
    # The jump operator is the identity on the odd-parity subspace spanned by
    # |01> and |10>, so stochastic clicks must not perturb the trajectory.
    ntrajs = 16
    omega = 1.3

    sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = 0.5 * omega * (dq.tensor(sx, sx) + dq.tensor(sy, sy))
    jump_ops = [-dq.tensor(sz, sz)]
    psi01 = dq.fock((2, 2), (0, 1))
    psi10 = dq.fock((2, 2), (1, 0))
    tsave = jnp.linspace(0.0, 1.2, 13)
    keys = jax.random.split(jax.random.key(1079), num=ntrajs)
    method = dq.method.Event(dtmax=1e-2)

    result = dq.jssesolve(
        H, jump_ops, psi01, tsave, keys=keys, method=method, nmaxclick=32
    )

    exact = (
        jnp.cos(omega * tsave)[:, None, None] * psi01.to_jax()
        - 1j * jnp.sin(omega * tsave)[:, None, None] * psi10.to_jax()
    )
    exact = dq.asqarray(exact, dims=(2, 2))

    infidelity = 1.0 - dq.overlap(exact, result.states)
    assert jnp.allclose(infidelity, 0.0, atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_deexcitation_bernoulli_statistics(atol=5e-2):
    # For H=0, L=sqrt(gamma) sigma_-, and psi0=|e>, each trajectory is either
    # excited or has emitted one jump, with P_e(t) ~ Bernoulli(exp(-gamma t)).
    ntrajs = 800
    gamma = 0.8
    tsave = jnp.linspace(0.0, 2.0, 21)
    keys = jax.random.split(jax.random.key(1080), num=ntrajs)
    exp_ops = [dq.excited_dm()]
    method = dq.method.Event(dtmax=2e-2)

    result = dq.jssesolve(
        dq.zeros_like(dq.sigmaz()),
        [jnp.sqrt(gamma) * dq.sigmam()],
        dq.excited(),
        tsave,
        keys=keys,
        exp_ops=exp_ops,
        method=method,
        nmaxclick=2,
    )

    excited_population = result.expects[:, 0, :].real
    expected_mean = jnp.exp(-gamma * tsave)
    expected_variance = expected_mean * (1.0 - expected_mean)

    assert jnp.allclose(excited_population * (1.0 - excited_population), 0.0, atol=1e-5)
    assert jnp.allclose(excited_population.mean(axis=0), expected_mean, atol=atol)
    assert jnp.allclose(excited_population.var(axis=0), expected_variance, atol=atol)
