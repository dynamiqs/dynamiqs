from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from dynamiqs.method import Method

from .systems.stochastic_system import StochasticSystem

# ── helpers ──────────────────────────────────────────────────────────────────


def infidelity_with_state(result, exact_states) -> Array:
    # infidelity with a pure target, for every trajectory and saved time;
    # normalized by the state norm/trace to be robust to Euler norm drift
    states = result.states.to_jax()  # (ntrajs, ntsave, n, k), k=1 (ket) or n (dm)
    exact = exact_states.to_jax()  # (ntsave, n, 1)
    bra = jnp.conj(jnp.swapaxes(exact, -1, -2))  # (ntsave, 1, n)
    if states.shape[-1] == 1:  # ket (SSE)
        overlap = (bra @ states)[..., 0, 0]  # (ntrajs, ntsave)
        bra_psi = jnp.conj(jnp.swapaxes(states, -1, -2))
        norm2 = (bra_psi @ states)[..., 0, 0].real
        return 1 - jnp.abs(overlap) ** 2 / norm2
    # density matrix (SME)
    num = (bra @ states @ exact)[..., 0, 0].real  # (ntrajs, ntsave)
    tr = jnp.trace(states, axis1=-2, axis2=-1).real
    return 1 - num / tr


def trajectory_norms(result) -> Array:
    # state norm per trajectory and time: <psi|psi> for kets, Tr[rho] for density
    # matrices. Probability conservation requires this to stay 1.
    states = result.states.to_jax()
    if states.shape[-1] == 1:
        bra = jnp.conj(jnp.swapaxes(states, -1, -2))
        return (bra @ states)[..., 0, 0].real
    return jnp.trace(states, axis1=-2, axis2=-1).real


def cross_trajectory_infidelity(result) -> Array:
    # infidelity between every trajectory and the first one, per saved time. With no
    # back-action all trajectories follow the same deterministic evolution, so this
    # is ~0; genuine back-action makes the trajectories differ.
    states = result.states.to_jax()  # (ntrajs, ntsave, n, k)
    if states.shape[-1] == 1:  # ket
        states = states / jnp.linalg.norm(states, axis=-2, keepdims=True)
        ref = states[0:1]
        overlap = jnp.abs((jnp.conj(jnp.swapaxes(ref, -1, -2)) @ states)[..., 0, 0])
        return 1 - overlap**2
    # density matrix: 1 - Tr[rho_0 rho_i] with unit-trace normalisation
    tr = jnp.trace(states, axis1=-2, axis2=-1).real[..., None, None]
    states = states / tr
    ref = states[0:1]
    return 1 - jnp.trace(ref @ states, axis1=-2, axis2=-1).real


def _keys(seed: int, ntrajs: int) -> Array:
    return jax.random.split(jax.random.key(seed), ntrajs)


# ── tester ───────────────────────────────────────────────────────────────────


class StochasticTester:
    """Hosts the stochastic-solver property tests, called from the per-solver test
    files (analogous to `IntegratorTester`). Each per-solver subclass sets `SOLVER`
    to the solver name; all references are analytical.
    """

    SOLVER: str

    def _test_convergence(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 2000,
        atol: float = 5e-3,
        atol_state: float = 1e-2,
        seed: int = 1,
    ):
        # averaging the trajectories recovers the analytical Lindblad evolution,
        # checked on both the expectation values and the full state (as in
        # IntegratorTester for the deterministic solvers)
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))

        true_expects = system.expects(system.tsave)
        assert jnp.allclose(result.mean_expects(), true_expects, atol=atol)

        true_states = system.states(system.tsave).todm().to_jax()
        errs = jnp.linalg.norm(
            result.mean_states().to_jax() - true_states, axis=(-2, -1)
        )
        assert jnp.all(errs < atol_state)

    def _test_no_backaction(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 20,
        atol: float = 1e-4,
        atol_norm: float = 2e-2,
        seed: int = 2,
    ):
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        exact = system.states(system.tsave)

        # direction: every trajectory follows the deterministic analytical state
        assert jnp.all(infidelity_with_state(result, exact) < atol)
        # scale: probability is conserved (Euler methods only to O(dt), the
        # norm-preserving Rouchon/Event methods to ~1e-6)
        assert jnp.allclose(trajectory_norms(result), 1.0, atol=atol_norm)
        # no back-action: all trajectories are identical up to integrator
        # floating-point error
        assert jnp.all(cross_trajectory_infidelity(result) < atol)

    def _test_backaction_is_detected(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 50,
        atol: float = 0.1,
        seed: int = 6,
    ):
        # control for _test_no_backaction: genuine back-action makes trajectories
        # deviate from the analytical curve and from each other
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        exact = system.states(system.tsave)
        assert jnp.max(infidelity_with_state(result, exact)) > atol
        assert jnp.max(cross_trajectory_infidelity(result)) > atol

    def _test_jump_statistics(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 4000,
        seed: int = 5,
    ):
        # click count over [0, T) is Poisson(gamma T): mean = var = gamma T. The
        # tolerances are 5-sigma Monte Carlo confidence bounds (scale as 1/sqrt(N)).
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        nclicks = result.nclicks[..., 0]
        n = nclicks.shape[0]
        lam = system.poisson_lambda()
        sem = jnp.sqrt(lam / n)
        sev = jnp.sqrt((lam + 2 * lam**2) / n)
        assert jnp.abs(nclicks.mean() - lam) < 5 * sem
        assert jnp.abs(nclicks.var() - lam) < 5 * sev

    def _test_diffusive_statistics(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 4000,
        seed: int = 5,
    ):
        # time-averaged record I = <L+Ld> + dW/dt is Gaussian, with analytically
        # known mean and variance; 5-sigma Monte Carlo confidence bounds
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        samples = result.measurements[..., 0, :].reshape(-1)
        n = samples.shape[0]
        mean, var = system.record_mean(), system.record_variance()
        sem = jnp.sqrt(var / n)
        sev = var * jnp.sqrt(2 / n)
        assert jnp.abs(samples.mean() - mean) < 5 * sem
        assert jnp.abs(samples.var() - var) < 5 * sev

    def _test_bernoulli_statistics(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 4000,
        seed: int = 3,
    ):
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        pe = result.expects[..., 0, :].real  # (ntrajs, ntsave), exactly 0 or 1
        mean, var = pe.mean(axis=0), pe.var(axis=0)
        p = system.excited_population(system.tsave)
        # 5-sigma Bernoulli Monte Carlo bound; the floor absorbs the t=0 point (zero
        # variance) and the O(dt) fixed-step bias
        bound = 5 * jnp.sqrt(p * (1 - p) / ntrajs) + 1e-3
        assert jnp.all(jnp.abs(mean - p) < bound)
        assert jnp.all(jnp.abs(var - p * (1 - p)) < bound)

    def _test_reject_wrong_rate(
        self,
        system: StochasticSystem,
        method: Method,
        *,
        ntrajs: int = 4000,
        seed: int = 7,
    ):
        # control for _test_jump_statistics: the measured click rate rejects a wrong
        # rate (2 gamma instead of gamma)
        result = system.run(self.SOLVER, method, _keys(seed, ntrajs))
        nclicks = result.nclicks[..., 0]
        assert jnp.abs(nclicks.mean() - 2 * system.poisson_lambda()) > 5e-2
