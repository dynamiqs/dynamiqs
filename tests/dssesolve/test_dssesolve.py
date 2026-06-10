import pytest

from dynamiqs.method import EulerMaruyama, Rouchon1

from ..order import TEST_LONG
from ..stochastic_tester import StochasticTester
from ..systems import backaction_qubit, damped_oscillator, protected_subspace, qnd_qubit

DT = 1e-3
_METHODS = [EulerMaruyama(dt=DT), Rouchon1(dt=DT)]


@pytest.mark.run(order=TEST_LONG)
class TestDSSESolve(StochasticTester):
    SOLVER = 'dsse'

    @pytest.mark.parametrize('method', _METHODS)
    def test_convergence(self, method):
        self._test_convergence(damped_oscillator, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_no_backaction(self, method):
        self._test_no_backaction(protected_subspace, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_statistics(self, method):
        self._test_diffusive_statistics(qnd_qubit, method)

    @pytest.mark.parametrize('method', _METHODS)
    def test_backaction_is_detected(self, method):
        self._test_backaction_is_detected(backaction_qubit, method)
