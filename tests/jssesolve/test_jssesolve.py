import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs.method import EulerJump, Event, Rouchon2

from ..order import TEST_INSTANT, TEST_LONG
from ..stochastic_tester import StochasticTester, trajectory_norms
from ..systems import (
    backaction_qubit,
    damped_oscillator,
    decay_qubit,
    protected_subspace,
    qnd_qubit,
)

DT = 1e-3


def _event(smart_sampling: bool) -> Event:
    # same root finder setup as the other jssesolve tests
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    return Event(root_finder=root_finder, smart_sampling=smart_sampling)


_METHODS = [EulerJump(dt=DT), _event(False), _event(True)]


@pytest.mark.run(order=TEST_LONG)
class TestJSSESolve(StochasticTester):
    SOLVER = 'jsse'

    @pytest.mark.parametrize('method', _METHODS)
    def test_convergence(self, method):
        self._test_convergence(damped_oscillator, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_no_backaction(self, method):
        self._test_no_backaction(protected_subspace, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_statistics(self, method):
        self._test_jump_statistics(qnd_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_bernoulli_statistics(self, method):
        self._test_bernoulli_statistics(decay_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_backaction_is_detected(self, method):
        self._test_backaction_is_detected(backaction_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_reject_wrong_rate(self, method):
        self._test_reject_wrong_rate(qnd_qubit, method)


@pytest.mark.run(order=TEST_INSTANT)
def test_event_unsupported_noclick_method():
    # an adaptive `Rouchon2` must not be reported as a fixed step size method
    with pytest.raises(TypeError, match='but has type `Rouchon2`'):
        Event(noclick_method=Rouchon2(), dtmax=1e-2)


@pytest.mark.run(order=TEST_LONG)
def test_euler_jump_driven_qubit():
    # Driven qubit decaying from |g>: the no-click evolution changes <LdL>, and the
    # click probability is exactly zero at t=0.
    def solve(omega):
        H = 0.5 * omega * dq.sigmax()
        tsave = jnp.linspace(0.0, 1.0, 11)
        keys = jax.random.split(jax.random.key(0), 4)
        return dq.jssesolve(
            H,
            [dq.sigmam()],
            dq.ground(),
            tsave,
            keys,
            exp_ops=[dq.sigmaz()],
            method=EulerJump(dt=1e-2),
        )

    def mean_sz(omega):
        return solve(omega).expects[:, 0, -1].real.mean()

    assert jnp.allclose(trajectory_norms(solve(3.0)), 1.0, atol=1e-5)
    assert jnp.allclose(jax.grad(mean_sz)(3.0), jax.jacfwd(mean_sz)(3.0), rtol=1e-4)
