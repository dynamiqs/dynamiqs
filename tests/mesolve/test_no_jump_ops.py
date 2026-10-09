import jax
import pytest

from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_nojump_cavity, dia_nojump_cavity

# with no jump operators, the Lindbladian computes `-1j * H - 0.0`: we test each vector
# field computing it, with and without `assume_hermitian` (implicit methods use the
# latter) and the fused DIA Lindbladian


@pytest.mark.run(order=TEST_LONG)
class TestMESolveNoJumpOps(IntegratorTester):
    @pytest.mark.parametrize(
        'system', [dense_nojump_cavity, dia_nojump_cavity], ids=['dense', 'dia']
    )
    @pytest.mark.parametrize(
        'assume_hermitian', [True, False], ids=['hermitian', 'standard']
    )
    def test_correctness(self, system, assume_hermitian):
        self._test_correctness(system, Tsit5(), assume_hermitian=assume_hermitian)


@pytest.mark.run(order=TEST_LONG)
class TestMESolveNoJumpOpsGPUPaths(IntegratorTester):
    # on GPU, the DIA Lindbladian is fused: CI runs on CPU, so this test forces it
    @pytest.fixture(autouse=True)
    def gpu_paths(self, monkeypatch):
        monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
        jax.clear_caches()
        yield
        jax.clear_caches()

    def test_correctness(self):
        self._test_correctness(dia_nojump_cavity, Tsit5())
