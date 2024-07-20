import jax
import jax.numpy as jnp
import pytest

from dynamiqs import Options, propagator, sigmay, pwc, eye, constant, rand_herm
from dynamiqs.solver import Tsit5

from ..sesolve.closed_system import cavity, tdqubit
from ..solver_tester import SolverTester


class TestPropagator(SolverTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system, ysave_atol: float = 1e-4):
        params = system.params_default
        H = system.H(params)
        y0 = system.y0(params)
        propresult = propagator(H, system.tsave)
        true_ysave = system.states(system.tsave)
        prop_ysave = jnp.einsum("ijk,kd->ijd", propresult.propagator, y0)
        errs = jnp.linalg.norm(true_ysave - prop_ysave, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    @pytest.mark.parametrize('solver', [None, Tsit5()])
    @pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
    def test_correctness_complex(self, nH, save_states, solver, ysave_atol: float = 1e-3):
        H = constant(rand_herm(jax.random.PRNGKey(42), (*nH, 2, 2)))
        t = 10.0
        tsave = jnp.linspace(0.0, t, 3)
        options = Options(save_states=save_states)
        propresult = propagator(H, tsave, solver=solver, options=options).propagator
        if save_states:
            Hs = jnp.einsum("...ij,t->...tij", H.array, tsave)
            trueresult = jax.scipy.linalg.expm(-1j * Hs)
        else:
            trueresult = jax.scipy.linalg.expm(-1j * H.array * t)
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    @pytest.mark.parametrize('solver', [None, Tsit5()])
    def test_correctness_pwc(self, save_states, solver, ysave_atol: float = 1e-4):
        times = [0.0, 1.0, 2.0]
        values = [3.0, -2.0]
        array = sigmay()
        H = pwc(times, values, array)
        tsave = jnp.asarray([0.0, 0.5, 1.5, 2.0])
        options = Options(save_states=save_states)
        propresult = propagator(H, tsave, solver=solver, options=options).propagator
        U0 = eye(H.shape[0])
        U1 = jax.scipy.linalg.expm(-1j * H.array * 3.0 * 0.5)
        U2 = jax.scipy.linalg.expm(-1j * H.array * -2.0 * 0.5)
        if save_states:
            trueresult = jnp.stack((U0, U1, U2 @ U1 @ U1, U2 @ U2 @ U1 @ U1))
        else:
            trueresult = U2 @ U2 @ U1 @ U1
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)
