import pytest

from dynamiqs.method import EulerJump

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
_METHODS = [EulerJump(dt=DT)]


@pytest.mark.run(order=TEST_LONG)
class TestJSMESolve(StochasticTester):
    SOLVER = 'jsme'

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
