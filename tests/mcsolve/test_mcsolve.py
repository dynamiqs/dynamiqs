# Set an environment variable
# os.environ['EQX_ON_ERROR'] = 'breakpoint'
# os.environ['JAX_DISABLE_JIT'] = '1'

import jax.numpy as jnp
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs import (
    Options,
    basis,
    mcsolve,
    mesolve,
    sigmax,
    sigmaz,
    timecallable,
    tobra,
)
from dynamiqs.solver import Tsit5

from ..integrator_tester import IntegratorTester


class TestMCSolve(IntegratorTester):
    @pytest.mark.skip(reason='not working yet')
    @pytest.mark.parametrize('one_jump_only', [True])
    @pytest.mark.parametrize('amp', [0.0, 0.001, 0.1])
    @pytest.mark.parametrize('omega_d_frac', [0.5, 0.999])
    @pytest.mark.parametrize('kappa', [0.001, 0.1])
    def test_against_mesolve(
        self, amp, omega_d_frac, kappa, one_jump_only, ysave_atol=1e-3
    ):
        num_traj = 51
        options = Options(one_jump_only=one_jump_only, ntraj=num_traj)
        omega = 2.0 * jnp.pi * 1.0
        omega_d = omega_d_frac * omega
        amp = 2.0 * jnp.pi * amp
        tsave = jnp.linspace(0.0, 100.0, 81)
        jump_ops = [jnp.sqrt(2.0 * jnp.pi * kappa) * basis(2, 0) @ tobra(basis(2, 1))]
        exp_ops = [basis(2, 0) @ tobra(basis(2, 0)), basis(2, 1) @ tobra(basis(2, 1))]
        initial_states = [dq.basis(2, 0)]

        def H_func(t):
            return -0.5 * omega * sigmaz() + jnp.cos(omega_d * t) * amp * sigmax()

        H = timecallable(H_func)
        mcresult = mcsolve(
            H,
            jump_ops,
            initial_states,
            tsave,
            exp_ops=exp_ops,
            options=options,
            solver=Tsit5(max_steps=1_000_000),
            root_finder=optx.Chord(1e-5, 1e-5),
        )
        meresult = mesolve(
            H,
            jump_ops,
            initial_states,
            tsave,
            exp_ops=exp_ops,
            options=options,
            solver=Tsit5(max_steps=1_000_000),
        )
        errs = jnp.linalg.norm(meresult.expects - mcresult.expects, axis=(-2, -1))
        assert jnp.all(errs <= ysave_atol)
