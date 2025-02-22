import jax
import jax.numpy as jnp
import pytest

from dynamiqs import Options, constant, eye, pwc, random, sepropagator, sigmax, sigmay
from dynamiqs.method import Tsit5

from ..integrator_tester import IntegratorTester
from ..order import TEST_LONG
from ..sesolve.closed_system import dense_cavity, tdqubit


@pytest.mark.run(order=TEST_LONG)
class TestSEPropagator(IntegratorTester):
    @pytest.mark.parametrize('system', [dense_cavity, tdqubit])
    def test_correctness(self, system, ysave_atol: float = 1e-4):
        params = system.params_default
        H = system.H(params)
        y0 = system.y0(params)
        propresult = sepropagator(H, system.tsave)
        true_ysave = system.states(system.tsave).to_jax()
        prop_ysave = (propresult.propagators @ y0).to_jax()
        assert jnp.allclose(true_ysave, prop_ysave, atol=ysave_atol)

    @pytest.mark.parametrize('save_propagators', [True, False])
    @pytest.mark.parametrize('method', [None, Tsit5()])
    @pytest.mark.parametrize('nH', [(), (4,), (4, 5)])
    def test_correctness_complex(
        self, nH, save_propagators, method, ysave_atol: float = 3e-4
    ):
        H = constant(random.herm(jax.random.PRNGKey(42), (*nH, 2, 2)))
        tsave = jnp.linspace(1.0, 10.0, 3)
        options = Options(save_propagators=save_propagators, t0=0.0)
        propresult = sepropagator(H, tsave, method=method, options=options)
        propagators = propresult.propagators.to_jax()
        ts = tsave if save_propagators else jnp.asarray([10.0])
        Hts = jnp.einsum('...ij,t->...tij', H.qarray.to_jax(), ts)
        true_propagators = jax.scipy.linalg.expm(-1j * Hts)
        assert jnp.allclose(propagators, true_propagators, atol=ysave_atol)

    @pytest.mark.parametrize('save_propagators', [True, False])
    @pytest.mark.parametrize('method', [None, Tsit5()])
    def test_correctness_pwc(self, save_propagators, method, ysave_atol: float = 1e-4):
        times = [0.0, 1.0, 2.0]
        values = [3.0, -2.0]
        qarray = sigmay()
        H = pwc(times, values, qarray)
        tsave = jnp.asarray([0.5, 1.0, 2.0])
        options = Options(save_propagators=save_propagators)
        propresult = sepropagator(H, tsave, method=method, options=options)
        propagators = propresult.propagators.to_jax()
        U0 = eye(H.shape[-1]).to_jax()
        U1 = jax.scipy.linalg.expm(-1j * H.qarray.to_jax() * 3.0 * 0.5)
        U2 = jax.scipy.linalg.expm(-1j * H.qarray.to_jax() * -2.0 * 1.0)
        if save_propagators:
            true_propagators = jnp.stack((U0, U1, U2 @ U1))
        else:
            true_propagators = (U2 @ U1)[None]
        assert jnp.allclose(propagators, true_propagators, atol=ysave_atol)

    @pytest.mark.parametrize('save_propagators', [True, False])
    @pytest.mark.parametrize('method', [None, Tsit5()])
    def test_correctness_summed_pwc(
        self, save_propagators, method, ysave_atol: float = 1e-4
    ):
        times_1 = [0.0, 1.0, 2.0]
        times_2 = [0.0, 0.5, 1.0, 2.5]
        values_1 = [3.0, -2.0]
        values_2 = [4.0, -5.0, 1.0]
        H1 = pwc(times_1, values_1, sigmay())
        H2 = pwc(times_2, values_2, sigmax())
        H = H1 + H2
        tsave = jnp.asarray([0.5, 1.0, 2.5])
        options = Options(save_propagators=save_propagators)
        propresult = sepropagator(H, tsave, method=method, options=options)
        propagators = propresult.propagators.to_jax()
        H1_array = H1.qarray.to_jax()
        H2_array = H2.qarray.to_jax()
        U0 = eye(H.shape[-1]).to_jax()
        U1 = jax.scipy.linalg.expm(-1j * (H1_array * 3.0 + H2_array * (-5.0)) * 0.5)
        U2 = jax.scipy.linalg.expm(-1j * (H1_array * (-2.0) + H2_array * 1.0) * 1.0)
        U3 = jax.scipy.linalg.expm(-1j * H2_array * 1.0 * 0.5)
        if save_propagators:
            true_propagators = jnp.stack((U0, U1, U3 @ U2 @ U1))
        else:
            true_propagators = (U3 @ U2 @ U1)[None]
        assert jnp.allclose(propagators, true_propagators, atol=ysave_atol)
