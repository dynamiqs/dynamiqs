import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import BackwardCheckpointed, Direct, Forward, HigherOrder
from dynamiqs.method import (
    Dopri5,
    Dopri8,
    Euler,
    Kvaerno3,
    Kvaerno5,
    LowRank,
    Rouchon1,
    Tsit5,
)

# Methods that support `HigherOrder` (explicit Diffrax ODE methods) and methods that
# don't (implicit solvers, low-rank, Rouchon). These guard the `SUPPORTED_GRADIENT`
# gate without running a full integration system.
SUPPORTED = [Euler(dt=1e-3), Dopri5(), Dopri8(), Tsit5()]
UNSUPPORTED = [
    Kvaerno3(),
    Kvaerno5(),
    LowRank(rank=1, ode_method=Tsit5(), key=jax.random.PRNGKey(0)),
    Rouchon1(dt=1e-3),
]


@pytest.mark.parametrize('method', SUPPORTED)
def test_higher_order_supported(method):
    assert method.supports_gradient(HigherOrder())
    # explicit ODE methods keep supporting the first-order gradients too
    assert method.supports_gradient(Direct())
    assert method.supports_gradient(BackwardCheckpointed())
    assert method.supports_gradient(Forward())


@pytest.mark.parametrize('method', UNSUPPORTED)
def test_higher_order_unsupported(method):
    assert not method.supports_gradient(HigherOrder())
    # first-order gradients are still supported, only HigherOrder is rejected
    assert method.supports_gradient(Direct())
    with pytest.raises(
        ValueError,
        match=f'Method `{type(method).__name__}` does not support gradient'
        ' `HigherOrder`',
    ):
        method.assert_supports_gradient(HigherOrder())


def test_default_gradient_rejects_hessian():
    # regression guard: the default first-order path (RecursiveCheckpointAdjoint, a
    # custom_vjp) cannot be Hessian-differentiated; `HigherOrder` is what unlocks it.
    t = 0.7

    def loss(omega):
        result = dq.sesolve(
            0.5 * omega * dq.sigmax(),
            dq.fock(2, 0),
            jnp.array([0.0, t]),
            exp_ops=[dq.sigmaz()],
            method=Tsit5(rtol=1e-7, atol=1e-7),
            gradient=BackwardCheckpointed(),
            progress_meter=False,
        )
        return result.expects[0, -1].real

    # first-order gradient still works
    assert jnp.isfinite(jax.grad(loss)(1.3))
    # second-order differentiation is rejected
    with pytest.raises(TypeError):
        jax.hessian(loss)(1.3)
