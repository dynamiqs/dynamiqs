import jax.tree_util as jtu
import optimistix as optx
import pytest

from dynamiqs.method import EulerJump, Event

from ..order import TEST_LONG
from ..stochastic_tester import StochasticTester
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


def _click_stat_methods() -> list:
    # Event(smart_sampling=True) biases the raw per-trajectory click ensemble, so
    # the Poisson and Bernoulli click statistics fail (solver bug, dynamiqs#1113).
    # The trajectory-averaged tests still pass because mean_*() corrects for it.
    xfail = pytest.mark.xfail(
        reason='Event(smart_sampling=True) raw click-statistics bias, dynamiqs#1113',
        strict=True,
    )
    return [
        pytest.param(m, marks=xfail) if isinstance(m, Event) and m.smart_sampling else m
        for m in _METHODS
    ]


@pytest.mark.run(order=TEST_LONG)
class TestJSSESolve(StochasticTester):
    SOLVER = 'jsse'

    @pytest.mark.parametrize('method', _METHODS)
    def test_convergence(self, method):
        self._test_convergence(damped_oscillator, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_no_backaction(self, method):
        self._test_no_backaction(protected_subspace, method)

    @pytest.mark.parametrize('method', _click_stat_methods())
    def test_statistics(self, method):
        self._test_jump_statistics(qnd_qubit, method)

    @pytest.mark.parametrize('method', _click_stat_methods())
    def test_bernoulli_statistics(self, method):
        self._test_bernoulli_statistics(decay_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_backaction_is_detected(self, method):
        self._test_backaction_is_detected(backaction_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_reject_wrong_rate(self, method):
        self._test_reject_wrong_rate(qnd_qubit, method)
