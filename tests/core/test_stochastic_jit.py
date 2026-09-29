import equinox as eqx
import jax
import pytest

import dynamiqs as dq
from dynamiqs.method import EulerJump, EulerMaruyama, Rouchon1

from ..order import TEST_SHORT

# the stochastic solvers only accept a static `tsave`, passed as a tuple
dt = 0.1
tsave = (0.0, dt, 2 * dt)
keys = jax.random.split(jax.random.PRNGKey(0), 2)

H = dq.sigmax()
jump_ops = [dq.sigmam(), dq.sigmaz()]
psi0 = dq.basis(2, 0)
etas = [1.0, 0.0]  # one measured and one purely dissipative channel
thetas = [0.0, 0.0]

solvers = {
    'jssesolve': lambda: dq.jssesolve(
        H, jump_ops, psi0, tsave, keys, method=EulerJump(dt=dt)
    ),
    'dssesolve': lambda: dq.dssesolve(
        H, jump_ops, psi0, tsave, keys, method=EulerMaruyama(dt=dt)
    ),
    'jsmesolve': lambda: dq.jsmesolve(
        H, jump_ops, thetas, etas, psi0, tsave, keys, method=EulerJump(dt=dt)
    ),
    'dsmesolve': lambda: dq.dsmesolve(
        H, jump_ops, etas, psi0, tsave, keys, method=Rouchon1(dt=dt)
    ),
}


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('solver', solvers.keys())
def test_jit(solver):
    result = eqx.filter_jit(solvers[solver])()
    assert result.states.shape[:2] == (len(keys), len(tsave))
