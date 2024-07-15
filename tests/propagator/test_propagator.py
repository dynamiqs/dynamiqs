import pytest

import jax
import jax.numpy as jnp
from dynamiqs import propagator, sigmay, Options

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
        prop_ysave = jnp.einsum("ijk,kd->ijd", propresult.propagators, y0)
        errs = jnp.linalg.norm(true_ysave - prop_ysave, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)

    @pytest.mark.parametrize('save_states', [True, False])
    def test_correctness_complex(self, save_states, ysave_atol: float = 1e-4):
        H = sigmay()
        t = 10.0
        tsave = jnp.linspace(0.0, t, 3)
        options = Options(save_states=save_states)
        propresult = propagator(H, tsave, options=options).propagators
        if save_states:
            Hs = jnp.einsum("ij,t->tij", H, tsave)
            trueresult = jax.scipy.linalg.expm(-1j * Hs)
        else:
            trueresult = jax.scipy.linalg.expm(-1j * H * t)
        errs = jnp.linalg.norm(propresult - trueresult)
        assert jnp.all(errs <= ysave_atol)
