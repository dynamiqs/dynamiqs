"""Test that `save_extra` receives the correct current time as first argument.

The hook returns `(1 + t) * expect(op, y)`, compared against the same quantity computed
post-hoc from the saved states and `tsave`. This fails if `t` is stubbed to zero, off by
one, or inconsistent with the saved states. Every solver family is covered, because each
one calls `save()` from its own integration loop.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs.method import (
    DiffusiveMonteCarlo,
    Euler,
    EulerJump,
    EulerMaruyama,
    Event,
    Expm,
    JumpMonteCarlo,
    Rouchon1,
    Tsit5,
)

from ..order import TEST_SHORT
from ..systems import damped_oscillator, dense_cavity, dense_ocavity, otdqubit, tdqubit

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


def _event(smart_sampling: bool) -> Event:
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    return Event(root_finder=root_finder, smart_sampling=smart_sampling)


@pytest.mark.run(order=TEST_SHORT)
class TestSaveExtra:
    # Expm only supports constant or pwc Hamiltonians, hence the constant-H systems
    @pytest.mark.parametrize(
        ('system', 'method'),
        [
            (tdqubit, Tsit5()),  # diffrax adaptive step
            (tdqubit, Euler(dt=DT)),  # diffrax fixed step
            (dense_cavity, Expm()),  # expm scan
        ],
    )
    def test_sesolve(self, system, method):
        result = system.run(method, save_extra=_hook(system))
        _assert_extra_correct(system, result)

    @pytest.mark.parametrize(
        ('system', 'method'),
        [
            (otdqubit, Tsit5()),
            (otdqubit, Rouchon1(dt=DT)),
            (dense_ocavity, Expm()),
            # for the Monte Carlo methods the hook is applied per save point to the
            # mean states
            (otdqubit, JumpMonteCarlo(_keys(), EulerJump(dt=DT))),
            (otdqubit, DiffusiveMonteCarlo(_keys(), EulerMaruyama(dt=DT))),
        ],
    )
    def test_mesolve(self, system, method):
        result = system.run(method, save_extra=_hook(system))
        _assert_extra_correct(system, result)

    @pytest.mark.parametrize(
        ('solver', 'method'),
        [
            ('jsse', _event(smart_sampling=False)),  # event integrator
            ('jsse', _event(smart_sampling=True)),
            ('jsse', EulerJump(dt=DT)),  # fixed step stochastic integrator
            ('jsme', EulerJump(dt=DT)),
            ('dsse', EulerMaruyama(dt=DT)),
            ('dsme', Rouchon1(dt=DT)),
        ],
    )
    def test_stochastic(self, solver, method):
        system = damped_oscillator
        result = system.run(solver, method, _keys(), save_extra=_hook(system))
        _assert_extra_correct(system, result)

    @pytest.mark.parametrize('method', [Tsit5(), Expm()])
    def test_propagator(self, method):
        system = dense_ocavity
        H, Ls = system.H(system.params_default), system.Ls(system.params_default)
        save_extra = lambda t, U: (1 + t) * U.trace()
        results = [
            dq.sepropagator(H, system.tsave, method=method, save_extra=save_extra),
            dq.mepropagator(H, Ls, system.tsave, method=method, save_extra=save_extra),
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
