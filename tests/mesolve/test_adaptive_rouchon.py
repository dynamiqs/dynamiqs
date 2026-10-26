import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Direct
from dynamiqs.method import Rouchon2, Rouchon3

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..systems import dense_ocavity, otdqubit

# for speed we don't test all possible options:
# - normalize: set to True
# - skip system dia_ocavity


@pytest.mark.run(order=TEST_LONG)
class TestMESolveAdaptiveRouchon(IntegratorTester):
    @pytest.mark.parametrize('method_class', [Rouchon2, Rouchon3])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    def test_correctness(self, method_class, system):
        self._test_correctness(system, method_class())

    @pytest.mark.parametrize('method_class', [Rouchon2])
    @pytest.mark.parametrize('system', [dense_ocavity, otdqubit])
    @pytest.mark.parametrize('gradient', [Direct()])
    def test_gradient(self, method_class, system, gradient):
        self._test_gradient(system, method_class(), gradient)

    @pytest.mark.parametrize('method_class', [Rouchon2, Rouchon3])
    def test_pwc_hamiltonian(self, method_class):
        """Regression test: adaptive Rouchon with a PWC Hamiltonian must not crash.

        A PWC Hamiltonian introduces discontinuities, causing diffrax to wrap the
        step-size controller in a ClipStepSizeController. Previously,
        dataclasses.replace() on that wrapper raised a TypeError.
        """
        n = 2
        H = dq.pwc(jnp.array([0.0, 0.5, 1.0]), jnp.array([1.0, 2.0]), dq.sigmax())
        jump_ops = [0.1 * dq.sigmam()]
        rho0 = dq.fock_dm(n, 0)
        tsave = jnp.linspace(0.0, 1.0, 11)
        result = dq.mesolve(H, jump_ops, rho0, tsave, method=method_class())
        assert result.states.shape[-2:] == (n, n)
