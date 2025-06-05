import jax
import jax.numpy as jnp
import pytest

from dynamiqs import (
    Options,
    dag,
    eye,
    mepropagator,
    pwc,
    slindbladian,
    unvectorize,
    vectorize,
)

from ..integrator_tester import IntegratorTester
from ..mesolve.open_system import dense_ocavity
from ..order import TEST_LONG
from .mepropagator_utils import rand_mepropagator_args


@pytest.mark.run(order=TEST_LONG)
class TestMEPropagator(IntegratorTester):
    def test_correctness(self, ysave_atol: float = 1e-4):
        system = dense_ocavity
        params = system.params_default
        H = system.H(params)
        Ls = system.Ls(params)
        y0 = system.y0(params)
        rho0 = y0 @ dag(y0)
        rho0_vec = vectorize(rho0)
        propresult = mepropagator(H, Ls, system.tsave)
        propagators = propresult.propagators.to_jax()
        prop_ysave = unvectorize(propagators @ rho0_vec).to_jax()
        true_ysave = system.states(system.tsave).to_jax()
        assert jnp.allclose(prop_ysave, true_ysave, atol=ysave_atol)

    @pytest.mark.parametrize('save_propagators', [True, False])
    def test_correctness_pwc(self, save_propagators, ysave_atol: float = 1e-4):
        times = [0.0, 1.0, 2.0]
        values = [3.0, -2.0]
        _H, Ls = rand_mepropagator_args(2, (), [(), ()])
        H = pwc(times, values, _H)
        tsave = jnp.asarray([0.5, 1.0, 2.0])
        options = Options(save_propagators=save_propagators)
        propresult = mepropagator(H, Ls, tsave, options=options)
        propagators = propresult.propagators.to_jax()
        U0 = eye(H.shape[-1] ** 2).to_jax()
        lindbladian_1 = slindbladian(3.0 * H.qarray, Ls)
        lindbladian_2 = slindbladian(-2.0 * H.qarray, Ls)
        U1 = jax.scipy.linalg.expm(lindbladian_1.to_jax() * 0.5)
        U2 = jax.scipy.linalg.expm(lindbladian_2.to_jax() * 1.0)
        true_propagators = jnp.stack([U0, U1, U2 @ U1]) if save_propagators else U2 @ U1
        assert jnp.allclose(propagators, true_propagators, atol=ysave_atol)
