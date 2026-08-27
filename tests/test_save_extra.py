"""Test that `save_extra` receives the correct current time as first argument.

The hook returns `(1 + t) * expect(op, y)`, compared against the same quantity
computed post-hoc from the saved states and `tsave`. This fails if `t` is stubbed
to zero, off by one, or inconsistent with the saved states.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs.method import (
    DiffusiveMonteCarlo,
    Dopri5,
    Euler,
    EulerJump,
    EulerMaruyama,
    Event,
    Expm,
    JumpMonteCarlo,
    LowRank,
    Rouchon1,
    Rouchon2,
    Rouchon3,
    Tsit5,
)

N = 4
DT = 1e-3
tsave = jnp.linspace(0.0, 1.0, 6)

a = dq.destroy(N)
adag_a = a.dag() @ a
psi0 = dq.fock(N, 1)

# time-dependent Hamiltonian (a constant drive cannot distinguish a correct t from
# a wrong one combined with a compensating state error)
Htd = dq.modulated(lambda t: jnp.cos(2.0 * t), adag_a)
# piecewise constant Hamiltonian for the Expm method, with discontinuity times
# interleaving with tsave to catch a times[1:]/times[:-1] off-by-one
Hpwc = dq.pwc(
    jnp.asarray([0.0, 0.15, 0.55, 1.0]), jnp.asarray([1.0, -0.5, 2.0]), adag_a
)

jump_ops = [0.5 * a]


def save_extra(t, y):
    return (1 + t) * dq.expect(adag_a, y)


def expected_extra(result):
    return (1 + tsave) * dq.expect(adag_a, result.states)


def assert_extra_correct(result, *, atol=1e-6):
    assert jnp.allclose(result.extra, expected_extra(result), atol=atol)


@pytest.mark.parametrize(
    ('H', 'method'),
    [
        (Htd, Tsit5()),
        (Htd, Dopri5()),
        (Htd, Euler(dt=DT)),
        (Htd, Rouchon1(dt=DT)),
        (Htd, Rouchon2(dt=DT)),
        (Htd, Rouchon3(dt=DT)),
        (Hpwc, Expm()),
    ],
)
def test_mesolve(H, method):
    result = dq.mesolve(H, jump_ops, psi0, tsave, method=method, save_extra=save_extra)
    assert_extra_correct(result)


@pytest.mark.parametrize(
    ('H', 'method'), [(Htd, Tsit5()), (Htd, Euler(dt=DT)), (Hpwc, Expm())]
)
def test_sesolve(H, method):
    result = dq.sesolve(H, psi0, tsave, method=method, save_extra=save_extra)
    assert_extra_correct(result)


def save_extra_propagator(t, U):
    return (1 + t) * U.trace()


@pytest.mark.parametrize('method', [Tsit5(), Expm()])
def test_sepropagator(method):
    result = dq.sepropagator(
        Hpwc, tsave, method=method, save_extra=save_extra_propagator
    )
    expected = (1 + tsave) * result.propagators.trace()
    assert jnp.allclose(result.extra, expected, atol=1e-6)


@pytest.mark.parametrize('method', [Tsit5(), Expm()])
def test_mepropagator(method):
    result = dq.mepropagator(
        Hpwc, jump_ops, tsave, method=method, save_extra=save_extra_propagator
    )
    expected = (1 + tsave) * result.propagators.trace()
    assert jnp.allclose(result.extra, expected, atol=1e-6)


def _event(smart_sampling: bool) -> Event:
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    return Event(root_finder=root_finder, smart_sampling=smart_sampling)


@pytest.mark.parametrize('method', [_event(False), _event(True), EulerJump(dt=DT)])
def test_jssesolve(method):
    keys = jax.random.split(jax.random.key(42), 5)
    result = dq.jssesolve(
        Htd, jump_ops, psi0, tsave, keys, method=method, save_extra=save_extra
    )
    assert_extra_correct(result)


def test_jsmesolve():
    keys = jax.random.split(jax.random.key(42), 5)
    result = dq.jsmesolve(
        Htd,
        jump_ops,
        [0.0],
        [0.7],
        psi0,
        tsave,
        keys,
        method=EulerJump(dt=DT),
        save_extra=save_extra,
    )
    assert_extra_correct(result)


@pytest.mark.parametrize('method', [EulerMaruyama(dt=DT), Rouchon1(dt=DT)])
def test_dssesolve(method):
    keys = jax.random.split(jax.random.key(42), 5)
    result = dq.dssesolve(
        Htd, jump_ops, psi0, tsave, keys, method=method, save_extra=save_extra
    )
    assert_extra_correct(result)


@pytest.mark.parametrize('method', [EulerMaruyama(dt=DT), Rouchon1(dt=DT)])
def test_dsmesolve(method):
    keys = jax.random.split(jax.random.key(42), 5)
    result = dq.dsmesolve(
        Htd, jump_ops, [0.7], psi0, tsave, keys, method=method, save_extra=save_extra
    )
    assert_extra_correct(result)


def _montecarlo_methods():
    keys = jax.random.split(jax.random.key(42), 10)
    return [
        JumpMonteCarlo(keys, EulerJump(dt=DT)),
        DiffusiveMonteCarlo(keys, EulerMaruyama(dt=DT)),
    ]


@pytest.mark.parametrize('method', _montecarlo_methods())
def test_mesolve_montecarlo(method):
    # the hook is applied per save point to the mean states
    result = dq.mesolve(
        Htd, jump_ops, psi0, tsave, method=method, save_extra=save_extra
    )
    assert_extra_correct(result)


def _lowrank_method(**kwargs) -> LowRank:
    return LowRank(rank=N // 2, ode_method=Tsit5(), key=jax.random.PRNGKey(0), **kwargs)


def test_mesolve_lowrank():
    result = dq.mesolve(
        Htd, jump_ops, psi0, tsave, method=_lowrank_method(), save_extra=save_extra
    )
    assert_extra_correct(result)


def test_mesolve_lowrank_save_extra_low_rank():
    # with is_save_extra_low_rank=True the hook receives (t, m); check t only
    method = _lowrank_method(is_save_extra_low_rank=True)
    result = dq.mesolve(
        Htd,
        jump_ops,
        psi0,
        tsave,
        method=method,
        save_extra=lambda t, m: t,  # noqa: ARG005
    )
    assert jnp.allclose(result.extra, tsave, atol=1e-6)


def test_jit_and_gradient():
    # check that the new time plumbing compiles and lets gradients flow
    def loss(kappa):
        result = dq.mesolve(
            Htd,
            [jnp.sqrt(kappa) * a],
            psi0,
            tsave,
            method=Tsit5(),
            save_extra=save_extra,
        )
        return result.extra[-1].real

    grad = jax.jit(jax.grad(loss))(0.25)
    assert jnp.isfinite(grad)
