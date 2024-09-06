import jax
import jax.numpy as jnp
import pytest

from dynamiqs import Options, constant, eye, pwc, random, sepropagator, sigmax, sigmay
from dynamiqs.solver import Tsit5

from ..integrator_tester import IntegratorTester
from ..sesolve.closed_system import cavity, tdqubit


class TestSEPropagator(IntegratorTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system, ysave_atol: float = 1e-4):
        params = system.params_default
        H = system.H(params)
        y0 = system.y0(params)
        propresult = sepropagator(H, system.tsave)
        true_ysave = system.states(system.tsave)
        prop_ysave = jnp.einsum('ijk,kd->ijd', propresult.propagators, y0)
        errs = jnp.linalg.norm(true_ysave - prop_ysave, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    @pytest.mark.parametrize('solver', [None, Tsit5()])
    @pytest.mark.parametrize('nH', [(), (4,), (4, 5)])
    def test_correctness_complex(
        self, nH, save_states, solver, ysave_atol: float = 1e-3
    ):
        H = constant(random.herm(jax.random.PRNGKey(42), (*nH, 2, 2)))
        t = 10.0
        tsave = jnp.linspace(1.0, t, 3)
        options = Options(save_states=save_states, t0=0.0)
        propresult = sepropagator(H, tsave, solver=solver, options=options).propagators
        ts = tsave if save_states else jnp.asarray([t])
        Hts = jnp.einsum('...ij,t->...tij', H.array, ts)
        trueresult = jax.scipy.linalg.expm(-1j * Hts)
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    @pytest.mark.parametrize('solver', [None, Tsit5()])
    def test_correctness_pwc(self, save_states, solver, ysave_atol: float = 1e-4):
        times = [0.0, 1.0, 2.0]
        values = [3.0, -2.0]
        array = sigmay()
        H = pwc(times, values, array)
        tsave = jnp.asarray([0.5, 1.0, 2.0])
        options = Options(save_states=save_states)
        propresult = sepropagator(H, tsave, solver=solver, options=options).propagators
        U0 = eye(H.shape[-1])
        U1 = jax.scipy.linalg.expm(-1j * H.array * 3.0 * 0.5)
        U2 = jax.scipy.linalg.expm(-1j * H.array * -2.0 * 1.0)
        trueresult = jnp.stack((U0, U1, U2 @ U1)) if save_states else (U2 @ U1)[None]
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    @pytest.mark.parametrize('solver', [None, Tsit5()])
    def test_correctness_summed_pwc(
        self, save_states, solver, ysave_atol: float = 1e-4
    ):
        times_1 = [0.0, 1.0, 2.0]
        times_2 = [0.0, 0.5, 1.0, 2.5]
        values_1 = [3.0, -2.0]
        values_2 = [4.0, -5.0, 1.0]
        H_1 = pwc(times_1, values_1, sigmay())
        H_2 = pwc(times_2, values_2, sigmax())
        H = H_1 + H_2
        tsave = jnp.asarray([0.5, 1.0, 2.5])
        options = Options(save_states=save_states)
        propresult = sepropagator(H, tsave, solver=solver, options=options).propagators
        U0 = eye(H.shape[-1])
        U1 = jax.scipy.linalg.expm(-1j * (H_1.array * 3.0 + H_2.array * (-5.0)) * 0.5)
        U2 = jax.scipy.linalg.expm(-1j * (H_1.array * (-2.0) + H_2.array * 1.0) * 1.0)
        U3 = jax.scipy.linalg.expm(-1j * H_2.array * 1.0 * 0.5)
        if save_states:
            trueresult = jnp.stack((U0, U1, U3 @ U2 @ U1))
        else:
            trueresult = (U3 @ U2 @ U1)[None]
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)
