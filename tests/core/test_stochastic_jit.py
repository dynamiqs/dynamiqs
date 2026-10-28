import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.method import EulerJump, EulerMaruyama, Rouchon1

from ..order import TEST_SHORT

dt = 0.1
keys = jax.random.split(jax.random.PRNGKey(0), 2)

H = dq.sigmax()
jump_ops = [dq.sigmam(), dq.sigmaz()]
psi0 = dq.basis(2, 0)
etas = [1.0, 0.0]  # one measured and one purely dissipative channel
thetas = [0.0, 0.0]

solvers = {
    'jssesolve': lambda tsave: dq.jssesolve(
        H, jump_ops, psi0, tsave, keys, method=EulerJump(dt=dt)
    ),
    'dssesolve': lambda tsave: dq.dssesolve(
        H, jump_ops, psi0, tsave, keys, method=EulerMaruyama(dt=dt)
    ),
    'jsmesolve': lambda tsave: dq.jsmesolve(
        H, jump_ops, thetas, etas, psi0, tsave, keys, method=EulerJump(dt=dt)
    ),
    'dsmesolve': lambda tsave: dq.dsmesolve(
        H, jump_ops, etas, psi0, tsave, keys, method=Rouchon1(dt=dt)
    ),
}


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('solver', solvers.keys())
def test_jit(solver):
    # `tsave` is traced, and not linearly spaced
    tsave = jnp.array([0.0, dt, 3 * dt])
    result = jax.jit(solvers[solver])(tsave)
    assert result.states.shape[:2] == (len(keys), len(tsave))


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('solver', solvers.keys())
def test_trajectory_independent_of_tsave(solver):
    # the noise of each step only depends on its index, so saving at more times must
    # not change the trajectory
    coarse = solvers[solver](jnp.array([0.0, 20 * dt]))
    fine = solvers[solver](jnp.linspace(0.0, 20 * dt, 21))
    assert jnp.allclose(coarse.final_state.to_jax(), fine.final_state.to_jax())
    assert jnp.array_equal(coarse.infos.nsteps, fine.infos.nsteps)
