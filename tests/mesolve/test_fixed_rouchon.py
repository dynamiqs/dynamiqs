import pytest

from dynamiqs.gradient import Autograd, CheckpointAutograd, ForwardAutograd
from dynamiqs.method import Rouchon1, Rouchon2, Rouchon3

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from .open_system import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - exact_expm: set to False
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
    @pytest.mark.parametrize(
        'gradient', [Autograd(), CheckpointAutograd(), ForwardAutograd()]
    )
    def test_gradient(self, method_class, dt, system, gradient):
        method = method_class(dt=dt)
        self._test_gradient(system, method, gradient)
