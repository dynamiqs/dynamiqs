import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.method import Tsit5

from ..order import TEST_SHORT

# The Hamiltonians below are proportional to sigma_x at all times, so H(t) commutes
# with itself and the exact solution is exp(-i theta(t) sigma_x) |psi0> with
# theta(t) the integral of the modulation.

def square_wave(t):
    # square wave of period 1, discontinuous at every multiple of 0.5 and of vanishing
    # integral over each period
    return jnp.where(jnp.sin(2 * jnp.pi * t) >= 0, 1.0, -1.0)


def solve(H, tsave):
    psi0 = dq.fock(2, 0)
    return dq.sesolve(H, psi0, tsave, method=Tsit5(), progress_meter=False)


@pytest.mark.run(order=TEST_SHORT)
def test_declared_discontinuities_improve_adaptive_stepping():
    # both Hamiltonians define the exact same vector field, but only the first one
    # declares where it jumps. The square wave has a vanishing integral over each
    # period, so the exact state is |psi0> at every integer time.
    tsave = jnp.arange(6.0)
    disc_ts = 0.5 * jnp.arange(11)
    declared = solve(
        dq.modulated(square_wave, dq.sigmax(), discontinuity_ts=disc_ts), tsave
    )
    hidden = solve(dq.modulated(square_wave, dq.sigmax()), tsave)
    psi0 = dq.fock(2, 0)
    error = lambda result: jnp.abs(result.states.to_jax() - psi0.to_jax()).max()
    assert declared.infos.nrejected < 0.5 * hidden.infos.nrejected
    assert error(declared) < error(hidden)


@pytest.mark.run(order=TEST_SHORT)
def test_repeated_discontinuity_ts():
    # `discontinuity_ts` is sorted but not deduplicated, here because both terms of the
    # sum jump at the same times
    times = jnp.linspace(0.0, 1.0, 6)
    values = jnp.array([1.0, -3.0, 5.0, -2.0, 4.0])
    H = dq.pwc(times, values, dq.sigmax()) + dq.pwc(times, values, dq.sigmax())
    assert len(H.discontinuity_ts) == 2 * len(times)

    tsave = jnp.linspace(0.0, 1.0, 11)
    states = solve(H, tsave).states
    expected = solve(dq.pwc(times, 2 * values, dq.sigmax()), tsave).states
    assert jnp.allclose(states.to_jax(), expected.to_jax(), atol=1e-5)
