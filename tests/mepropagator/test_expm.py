import jax
import jax.numpy as jnp
import pytest

from dynamiqs import (
    Options,
    dag,
    eye,
    mepropagator,
    operator_to_vector,
    pwc,
    slindbladian,
    vector_to_operator,
)

from ..integrator_tester import IntegratorTester
from ..mesolve.open_system import ocavity
from .mepropagator_utils import rand_mepropagator_args


class TestMEPropagator(IntegratorTester):
    def test_correctness(self, ysave_atol: float = 1e-4):
        system = ocavity
        params = system.params_default
        H = system.H(params)
        Ls = system.Ls(params)
        y0 = system.y0(params)
        rho0 = y0 @ dag(y0)
        rho0_vec = operator_to_vector(rho0)
        propresult = mepropagator(H, Ls, system.tsave)
        true_ysave = system.states(system.tsave)
        prop_ysave = jnp.einsum('ijk,kd->ijd', propresult.propagators, rho0_vec)
        prop_ysave = vector_to_operator(prop_ysave)
        errs = jnp.linalg.norm(true_ysave - prop_ysave, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    def test_correctness_pwc(self, save_states, ysave_atol: float = 1e-4):
        times = [0.0, 1.0, 2.0]
        values = [3.0, -2.0]
        _H, Ls = rand_mepropagator_args(2, (), [(), ()])
        H = pwc(times, values, _H)
        tsave = jnp.asarray([0.5, 1.0, 2.0])
        options = Options(save_states=save_states)
        propresult = mepropagator(H, Ls, tsave, options=options).propagators
        U0 = eye(H.shape[-1] ** 2)
        lindbladian_1 = slindbladian(3.0 * H.array, Ls)
        lindbladian_2 = slindbladian(-2.0 * H.array, Ls)
        U1 = jax.scipy.linalg.expm(lindbladian_1 * 0.5)
        U2 = jax.scipy.linalg.expm(lindbladian_2 * 1.0)
        trueresult = jnp.stack((U0, U1, U2 @ U1)) if save_states else U2 @ U1
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)
