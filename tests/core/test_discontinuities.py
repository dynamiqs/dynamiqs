import jax
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


def solve(H, tsave, **kwargs):
    psi0 = dq.fock(2, 0)
    return dq.sesolve(H, psi0, tsave, method=Tsit5(), progress_meter=False, **kwargs)


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


@pytest.mark.run(order=TEST_SHORT)
def test_gradient_wrt_discontinuity_time():
    # the vector field depends on the switching time tau only through where it jumps,
    # so the derivative vanishes unless the steps are clipped to tau
    v0, v1 = 1.3, -0.7

    def population(tau):
        H = dq.pwc(jnp.stack([0.0, tau, 1.0]), jnp.array([v0, v1]), dq.sigmax())
        result = solve(H, jnp.array([0.0, 1.0]), gradient=dq.gradient.Forward())
        return jnp.abs(result.states[-1].to_jax()[0, 0]) ** 2

    # |⟨0|psi(1)⟩|² = cos²(theta) with theta = v0 tau + v1 (1 - tau)
    tau = 0.37
    theta = v0 * tau + v1 * (1 - tau)
    expected = -jnp.sin(2 * theta) * (v0 - v1)
    assert jnp.allclose(jax.jacfwd(population)(tau), expected, rtol=1e-3)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize(
    'progress_meter', [dq.TqdmProgressMeter(), dq.TextProgressMeter()]
)
def test_gradient_wrt_discontinuity_time_with_progress_meter(progress_meter):
    # the progress meters display the solver's time, which depends on tau: the
    # derivative must go through them as without a meter
    def population(tau, meter):
        H = dq.pwc(jnp.stack([0.0, tau, 1.0]), jnp.array([1.3, -0.7]), dq.sigmax())
        result = dq.sesolve(
            H,
            dq.fock(2, 0),
            jnp.array([0.0, 1.0]),
            method=Tsit5(),
            gradient=dq.gradient.Forward(),
            progress_meter=meter,
        )
        return jnp.abs(result.states[-1].to_jax()[0, 0]) ** 2

    expected = jax.jacfwd(lambda tau: population(tau, False))(0.37)
    assert jnp.allclose(
        jax.jacfwd(lambda tau: population(tau, progress_meter))(0.37), expected
    )
