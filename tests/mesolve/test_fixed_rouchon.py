import pytest

from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward
from dynamiqs.method import Rouchon1, Rouchon2, Rouchon3

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_driven_ocavity, dense_nojump_cavity, dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveFixedRouchon(IntegratorTester):
    @pytest.mark.parametrize(
        ('method_class', 'dt'), [(Rouchon1, 1e-4), (Rouchon2, 1e-3), (Rouchon3, 1e-2)]
    )
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, method_class, dt, system):
        method = method_class(dt=dt)
        self._test_correctness(system, method)

    @pytest.mark.parametrize(
        ('method_class', 'dt'), [(Rouchon1, 1e-4), (Rouchon2, 1e-3), (Rouchon3, 1e-2)]
    )
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct(), BackwardCheckpointed(), Forward()])
    def test_gradient(self, method_class, dt, system, gradient):
        method = method_class(dt=dt)
        self._test_gradient(system, method, gradient)

    # Rouchon1 overrides the Kraus map that Rouchon2 and Rouchon3 share
    @pytest.mark.parametrize(
        ('method_class', 'dt'), [(Rouchon1, 1e-4), (Rouchon2, 1e-3)]
    )
    def test_correctness_no_jump_ops(self, method_class, dt):
        self._test_correctness(dense_nojump_cavity, method_class(dt=dt))

    # the drive makes sum_k Mk^† @ Mk non-diagonal, see
    # `unrotated_cholesky_normalize()`; the larger amplitude needs smaller steps than
    # `test_correctness`
    @pytest.mark.parametrize(
        ('method_class', 'dt'), [(Rouchon2, 5e-4), (Rouchon3, 2e-3)]
    )
    def test_correctness_driven(self, method_class, dt):
        self._test_correctness(dense_driven_ocavity, method_class(dt=dt))
