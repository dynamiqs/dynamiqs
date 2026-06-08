import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG
from .utils import (
    JUMP_SOLVERS,
    SOLVERS,
    backaction_system,
    cross_trajectory_infidelity,
    decay_system,
    infidelity_with_state,
    protected_subspace_state,
    protected_subspace_system,
    qnd_system,
    trajectory_norms,
)

# ── ensemble convergence to mesolve ──────────────────────────────────────────


# this is a Monte Carlo estimate of the ensemble average, so the error is
# dominated by sampling fluctuation rather than a fixed bias: at 2000 trajectories
# the max deviation of the diffusive solvers ranges over ~0.008-0.028 depending on
# the seed (the jump solvers sit around ~0.004-0.006). 3e-2 is therefore the
# tightest tolerance that stays robust across RNG seeds; the deterministic
# trajectory-level test (test_no_backaction) is tightened to 1e-4 instead.
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', list(SOLVERS))
def test_convergence_to_mesolve(solver, atol=3e-2):
    n = 4
    a = dq.destroy(n)
    H = 0.1 * a.dag() @ a + 0.4 * (a + a.dag())
    jump_ops = [a, 0.3 * a.dag()]
    psi0 = dq.coherent(n, 0.5)
    exp_ops = [a.dag() @ a, a + a.dag()]
    tsave = jnp.linspace(0.0, 1.0, 6)

    keys = jax.random.split(jax.random.key(1), num=2000)
    sresult = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, exp_ops)
    meresult = dq.mesolve(
        H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=None
    )

    # averaging the trajectories recovers the Lindblad evolution, both for the
    # expectation values and for the states
    assert jnp.allclose(meresult.expects, sresult.mean_expects(), atol=atol)
    assert jnp.allclose(
        meresult.states.to_jax(), sresult.mean_states().to_jax(), atol=atol
    )


# ── no measurement back-action (protected subspace) ──────────────────────────


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', list(SOLVERS))
def test_no_backaction(solver, atol=1e-4, atol_norm=2e-2):
    omega = 1.0
    H, jump_ops, psi0 = protected_subspace_system(omega)
    tsave = jnp.linspace(0.0, 1.0, 11)

    keys = jax.random.split(jax.random.key(2), num=20)
    result = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, None)

    exact = dq.stack([protected_subspace_state(t.item(), omega) for t in tsave])

    # direction: every trajectory follows the deterministic analytical state
    assert jnp.all(infidelity_with_state(result, exact) < atol)

    # scale: probability is conserved (fixed-step Euler methods conserve the norm
    # only to O(dt) ~ 1e-2, the norm-preserving Rouchon/Event methods to ~1e-6)
    assert jnp.allclose(trajectory_norms(result), 1.0, atol=atol_norm)

    # no back-action: all trajectories follow the same deterministic evolution, so
    # they differ only at the integrator's floating-point level — a spurious
    # back-action would instead make them diverge from each other
    assert jnp.all(cross_trajectory_infidelity(result) < atol)


# ── statistics: common QND problem, one per unraveling ───────────────────────

# jsse with Event(smart_sampling=True) returns a biased *raw* click ensemble (the
# no-click trajectory is sampled separately), so the per-trajectory click
# statistics are wrong: the Poisson click count is ~7 sigma high and the
# Bernoulli population is off. This is a solver bug, tracked in dynamiqs#1113;
# the tests are correct to catch it, so the affected parametrizations are marked
# xfail (strict, so a future fix surfaces as an xpass). mean_expects()/
# mean_states() are corrected for smart sampling, so the convergence and
# no-back-action tests still pass.
_CLICK_STATS_XFAIL = {'jsse_event_smart'}


def _click_stats_param(name):
    if name in _CLICK_STATS_XFAIL:
        return pytest.param(
            name,
            marks=pytest.mark.xfail(
                reason='Event(smart_sampling=True) raw click-statistics bias, '
                'dynamiqs#1113',
                strict=True,
            ),
        )
    return name


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', [_click_stats_param(s) for s in SOLVERS])
def test_statistics(solver):
    # single common physical problem for all four solvers: a QND measurement of
    # sz on a sz eigenstate (H = 0, L = sqrt(gamma) sz, psi0 = |e>). The state is
    # a fixed point, so the only stochasticity is in the measurement record,
    # whose first and second moments are known analytically for each unraveling.
    gamma = 1.0
    H, jump_ops, psi0 = qnd_system(gamma)
    tsave = jnp.linspace(0.0, 1.0, 11)
    total_time = tsave[-1].item()

    keys = jax.random.split(jax.random.key(5), num=4000)
    result = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, None)

    # tolerances are 5-sigma Monte Carlo confidence bounds, which make the
    # expected scaling with the number of samples (as 1/sqrt(N)) explicit
    if solver in JUMP_SOLVERS:
        # the click count over [0, T) is Poisson(gamma T): mean = var = gamma T
        nclicks = result.nclicks[..., 0]  # (ntrajs,)
        n = nclicks.shape[0]
        lam = gamma * total_time
        sem = jnp.sqrt(lam / n)  # standard error of the mean
        sev = jnp.sqrt((lam + 2 * lam**2) / n)  # standard error of the variance
        assert jnp.abs(nclicks.mean() - lam) < 5 * sem
        assert jnp.abs(nclicks.var() - lam) < 5 * sev
    else:
        # the time-averaged record I = <L+Ld> + dW/dt is Gaussian, with mean
        # 2 sqrt(gamma) and variance 1/dt_save (independent of eta)
        dt_save = (tsave[1] - tsave[0]).item()
        samples = result.measurements[..., 0, :].reshape(-1)  # pool trajs & ints
        n = samples.shape[0]
        mean, var = 2 * jnp.sqrt(gamma), 1 / dt_save
        sem = jnp.sqrt(var / n)  # standard error of the mean
        sev = var * jnp.sqrt(2 / n)  # Gaussian standard error of the variance
        assert jnp.abs(samples.mean() - mean) < 5 * sem
        assert jnp.abs(samples.var() - var) < 5 * sev


# ── extra jump statistics: Bernoulli de-excitation ───────────────────────────


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', [_click_stats_param(s) for s in JUMP_SOLVERS])
def test_jump_bernoulli_statistics(solver):
    gamma = 1.0
    H, jump_ops, psi0 = decay_system(gamma)
    Pe = dq.excited().todm()
    tsave = jnp.linspace(0.0, 1.0, 11)

    ntrajs = 4000
    keys = jax.random.split(jax.random.key(3), num=ntrajs)
    result = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, [Pe])

    # excited population is exactly 0 or 1 on each trajectory, Bernoulli(e^{-gt})
    pe = result.expects[..., 0, :].real  # (ntrajs, ntsave)
    mean = pe.mean(axis=0)
    var = pe.var(axis=0)

    p = jnp.exp(-gamma * tsave)
    # 5-sigma Bernoulli Monte Carlo bound (scales as 1/sqrt(ntrajs)); the small
    # floor absorbs the t=0 point (zero variance) and the O(dt) fixed-step bias
    bound = 5 * jnp.sqrt(p * (1 - p) / ntrajs) + 1e-3
    assert jnp.all(jnp.abs(mean - p) < bound)
    assert jnp.all(jnp.abs(var - p * (1 - p)) < bound)


# ── negative controls (discriminating power) ─────────────────────────────────


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', list(SOLVERS))
def test_backaction_is_detected(solver, atol=0.1):
    # control for test_no_backaction: with a loss operator that is not the
    # identity on the subspace, genuine back-action makes trajectories deviate
    # from the deterministic curve AND from each other, so both no-back-action
    # assertions must fail
    omega = 1.0
    H, jump_ops, psi0 = backaction_system(omega)
    tsave = jnp.linspace(0.0, 1.0, 11)

    keys = jax.random.split(jax.random.key(6), num=50)
    result = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, None)

    exact = dq.stack([protected_subspace_state(t.item(), omega) for t in tsave])
    assert jnp.max(infidelity_with_state(result, exact)) > atol
    assert jnp.max(cross_trajectory_infidelity(result)) > atol


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('solver', JUMP_SOLVERS)
def test_statistics_reject_wrong_rate(solver):
    # control for test_statistics: the measured click statistics must reject a
    # wrong jump rate (2 gamma instead of gamma)
    gamma = 1.0
    H, jump_ops, psi0 = qnd_system(gamma)
    tsave = jnp.linspace(0.0, 1.0, 11)
    total_time = tsave[-1].item()

    keys = jax.random.split(jax.random.key(7), num=4000)
    result = SOLVERS[solver](H, jump_ops, psi0, tsave, keys, None)

    nclicks = result.nclicks[..., 0]
    assert jnp.abs(nclicks.mean() - 2 * gamma * total_time) > 5e-2
