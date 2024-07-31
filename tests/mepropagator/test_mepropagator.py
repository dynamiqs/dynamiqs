import os

os.environ['JAX_DISABLE_JIT'] = '1'

import jax.numpy as jnp
import pytest

from dynamiqs import dag, mepropagator, operator_to_vector, vector_to_operator
from dynamiqs.solver import Expm, Tsit5

from ..integrator_tester import IntegratorTester
from ..mesolve.open_system import ocavity, otdqubit


class TestMEPropagator(IntegratorTester):
    @pytest.mark.parametrize('system', [ocavity, otdqubit])
    def test_correctness(self, system, ysave_atol: float = 1e-4):
        params = system.params_default
        H = system.H(params)
        Ls = system.Ls(params)
        y0 = system.y0(params)
        rho0 = y0 @ dag(y0)
        rho0_vec = operator_to_vector(rho0)
        propresult = mepropagator(H, Ls, system.tsave, solver=Tsit5())
        true_ysave = system.states(system.tsave)
        prop_ysave = jnp.einsum('ijk,kd->ijd', propresult.propagators, rho0_vec)
        prop_ysave = vector_to_operator(prop_ysave)
        errs = jnp.linalg.norm(true_ysave - prop_ysave, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)

    def test_expm_vs_tsit5(self, ysave_atol=1e-4):
        system = ocavity
        params = system.params_default
        H = system.H(params)
        Ls = system.Ls(params)
        prop_expm = mepropagator(H, Ls, system.tsave, solver=Expm())
        prop_tsit5 = mepropagator(H, Ls, system.tsave, solver=Tsit5())
        errs = jnp.linalg.norm(
            prop_expm.propagators - prop_tsit5.propagators, axis=(-2, -1)
        )
        assert jnp.all(errs <= ysave_atol)
