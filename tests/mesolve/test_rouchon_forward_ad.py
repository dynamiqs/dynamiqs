import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Forward
from dynamiqs.method import Rouchon1, Rouchon2, Rouchon3, Tsit5

from ..order import TEST_SHORT

# Regression tests for forward-mode differentiation with respect to a parameter
# entering the time grid. Adaptive Rouchon used to force a step at every time in
# `tsave` to compensate for a first-order interpolation. This made every save-time
# interpolation interval degenerate, and `jacfwd` returned a Jacobian off by orders of
# magnitude (and of the wrong sign) instead of raising.

METHODS = [
    Rouchon1(dt=1e-4),
    Rouchon2(dt=1e-3),
    Rouchon3(dt=1e-2),
    Rouchon2(),
    Rouchon3(),
]

# The Hamiltonian must be time-dependent and `t0` must be pinned explicitly: with a
# constant Hamiltonian the dynamics is invariant under time translation, so shifting
# `tsave` and `t0` together leaves the states unchanged and these tests pass even with
# a broken tangent path. Do not simplify this to a constant Hamiltonian.
H = dq.modulated(lambda t: jnp.cos(3.0 * t), dq.sigmax())
jump_ops = [jnp.sqrt(0.3) * dq.sigmam()]
rho0 = dq.fock_dm(2, 0)
tsave = jnp.linspace(0.1, 1.0, 11)


def loss(shift, method, gradient=None, progress_meter=False):
    # expectation value at the last save time, as a function of a shift of `tsave`
    result = dq.mesolve(
        H,
        jump_ops,
        rho0,
        tsave + shift,
        exp_ops=[dq.sigmaz()],
        method=method,
        gradient=gradient,
        progress_meter=progress_meter,
        t0=0.0,
    )
    return result.expects[0, -1].real


def reference_derivative():
    # central difference with a tight ODE method, i.e. neither the Rouchon schemes nor
    # automatic differentiation. `rtol=2e-2` below is set by float32 rounding
    # accumulated over the 10000 steps of `Rouchon1(dt=1e-4)`; the broken tangent path
    # was off by 12 orders of magnitude, so the window is not the point here.
    h = 3e-3
    method = Tsit5(rtol=1e-8, atol=1e-8)
    return (loss(h, method) - loss(-h, method)) / (2 * h)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('method', METHODS)
def test_forward_gradient_tsave_shift(method):
    jac = jax.jacfwd(loss)(0.0, method, Forward())
    assert jnp.allclose(jac, reference_derivative(), rtol=2e-2)


@pytest.mark.run(order=TEST_SHORT)
def test_forward_gradient_with_progress_meter():
    # Diffrax reduces the progress over batch dimensions with `unvmap_max`, which has
    # no JVP rule, so a progress meter used to raise a `NotImplementedError` as soon as
    # the time grid carried a tangent. `_ForwardModeProgressMeter` drops that tangent.
    jac = jax.jacfwd(loss)(0.0, Rouchon2(), Forward(), True)
    assert jnp.allclose(jac, reference_derivative(), rtol=2e-2)
