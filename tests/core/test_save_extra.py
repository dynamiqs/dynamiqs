"""Test that `save_extra` receives the correct current time as first argument.

The hook returns `(1 + t) * expect(op, y)`, compared against the same quantity computed
post-hoc from the saved states and `tsave`. This fails if `t` is stubbed to zero, off by
one, or inconsistent with the saved states. All integrators share the same `save_extra`
pipeline, so we only keep one representative combination per solver family.
"""

import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.method import EulerJump, Expm, Tsit5

from ..order import TEST_SHORT
from ..systems import damped_oscillator, dense_cavity, dense_ocavity, otdqubit

DT = 1e-2
NTRAJS = 3


def _hook(system):
    op = dq.number(system.n)
    return lambda t, y: (1 + t) * dq.expect(op, y)


def _assert_extra_correct(system, result, *, atol=1e-6):
    op = dq.number(system.n)
    expected = (1 + jnp.asarray(system.tsave)) * dq.expect(op, result.states)
    assert jnp.allclose(result.extra, expected, atol=atol)


def _keys(ntrajs=NTRAJS):
    return jax.random.split(jax.random.key(42), ntrajs)


@pytest.mark.run(order=TEST_SHORT)
class TestSaveExtra:
    def test_sesolve(self):
        # the expm scan builds the save times itself, unlike the diffrax integrators
        # which get them from diffrax, hence the `Expm` method here
        system = dense_cavity
        result = system.run(Expm(), save_extra=_hook(system))
        _assert_extra_correct(system, result)

    def test_mesolve(self):
        system = otdqubit
        result = system.run(Tsit5(), save_extra=_hook(system))
        _assert_extra_correct(system, result)

    def test_stochastic(self):
        # the fixed step stochastic integrator carries the time in its own scan
        system = damped_oscillator
        result = system.run('jsse', EulerJump(dt=DT), _keys(), save_extra=_hook(system))
        _assert_extra_correct(system, result)

    def test_propagator(self):
        system = dense_ocavity
        H, Ls = system.H(system.params_default), system.Ls(system.params_default)
        save_extra = lambda t, U: (1 + t) * U.trace()
        results = [
            dq.sepropagator(H, system.tsave, method=Tsit5(), save_extra=save_extra),
            dq.mepropagator(H, Ls, system.tsave, method=Tsit5(), save_extra=save_extra),
        ]
        for result in results:
            expected = (1 + jnp.asarray(system.tsave)) * result.propagators.trace()
            assert jnp.allclose(result.extra, expected, atol=1e-6)

    def test_jit_and_gradient(self):
        # check that the new time plumbing compiles and lets gradients flow
        system = otdqubit

        def loss(eps):
            params = system.params_default._replace(eps=eps)
            result = system.run(Tsit5(), params=params, save_extra=_hook(system))
            return result.extra[-1].real

        grad = jax.jit(jax.grad(loss))(system.params_default.eps)
        assert jnp.isfinite(grad)
