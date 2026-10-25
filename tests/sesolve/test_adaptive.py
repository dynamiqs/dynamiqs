import pytest

from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.method import Kvaerno5, Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import (
    dense_cavity,
    dense_composite_cavity,
    dia_cavity,
    dia_composite_cavity,
    tdqubit,
)

# we only test Tsit5 to keep the unit test suite fast, and Kvaerno5 on the
# time-dependent qubit for implicit methods


@pytest.mark.run(order=TEST_LONG)
class TestSESolveAdaptive(IntegratorTester):
    @pytest.mark.parametrize(
        'system',
        [
            dense_cavity,
            dia_cavity,
            tdqubit,
            dense_composite_cavity,
            dia_composite_cavity,
        ],
    )
    def test_correctness(self, system):
        self._test_correctness(system, Tsit5())

    @pytest.mark.parametrize('system', [dense_cavity, dia_cavity, tdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, system, gradient):
        self._test_gradient(system, Tsit5(), gradient)

    def test_correctness_implicit(self):
        self._test_correctness(tdqubit, Kvaerno5())
