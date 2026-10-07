import diffrax as dx
import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core.diffrax_integrator import (
    MESolveDiffraxIntegrator,
    _saveat,
    call_diffeqsolve,
)
from dynamiqs.integrators.core.unrolled_solvers import _UnrolledERK

from ..order import TEST_SHORT

# the saves are split into real arrays on GPU only, so these tests force it on CPU,
# with the other GPU-only paths off: the split then leaves every output unchanged

n = 6
a = dq.destroy(n)
tsave = jnp.linspace(0.0, 1.0, 4)


def _H(amplitude):
    return a.dag() @ a + dq.modulated(lambda t: amplitude * jnp.cos(t), a + a.dag())


# every solver whose saves go through `_saveat`
SOLVES = {
    'mesolve': lambda amplitude: dq.mesolve(
        _H(amplitude),
        [0.5 * a],
        dq.coherent(n, 0.5),
        tsave,
        exp_ops=[a],
        progress_meter=False,
    ),
    'sesolve': lambda amplitude: dq.sesolve(
        _H(amplitude), dq.coherent(n, 0.5), tsave, exp_ops=[a], progress_meter=False
    ),
    'sepropagator': lambda amplitude: dq.sepropagator(
        _H(amplitude), tsave, progress_meter=False
    ),
    'mesolve_lowrank': lambda amplitude: dq.mesolve(
        _H(amplitude),
        [0.5 * a],
        dq.coherent(n, 0.5),
        tsave,
        exp_ops=[a],
        method=dq.method.LowRank(2, key=jax.random.PRNGKey(0)),
        progress_meter=False,
    ),
    'jssesolve_event': lambda amplitude: dq.jssesolve(
        _H(amplitude),
        [0.5 * a],
        dq.coherent(n, 0.5),
        tsave,
        jax.random.split(jax.random.PRNGKey(0), 2),
        exp_ops=[a],
        method=dq.method.Event(dtmax=0.1),
    ),
}


def _force_split_saves(monkeypatch):
    # the unrolled Runge-Kutta stages and the fused DIA Lindbladian (also GPU-only)
    # change the round-off, enough to change the steps of an adaptive solve, so keep
    # Diffrax's stages and the products here
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(_UnrolledERK, '_unrolled', lambda *args: False)  # noqa: ARG005
    monkeypatch.setattr(
        MESolveDiffraxIntegrator,
        '_are_operators_sparsedia',
        lambda self: False,  # noqa: ARG005
    )
    jax.clear_caches()


def _assert_match(split, reference):
    for x, y in zip(jax.tree.leaves(split), jax.tree.leaves(reference), strict=True):
        assert x.dtype == y.dtype
        assert jnp.array_equal(x, y, equal_nan=True)


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('solver', SOLVES)
def test_real_saves_match(monkeypatch, solver):
    # a batch of driven cavities, with states and expectation values saved
    solve = lambda: jax.vmap(SOLVES[solver])(jnp.array([0.5, 1.0]))
    reference = solve()
    _force_split_saves(monkeypatch)
    _assert_match(solve(), reference)


@pytest.mark.run(order=TEST_SHORT)
def test_saves_are_split(monkeypatch):
    # the tests above pass whether or not the saves are split: check that they are
    _force_split_saves(monkeypatch)
    y0 = dq.coherent(n, 0.5).to_jax()
    saveat, _ = _saveat(tsave, lambda t, y: {'y': y, 'norm': jnp.abs(y).sum()}, y0)  # noqa: ARG005
    for sub in saveat.subs:
        for leaf in jax.tree.leaves(sub.fn(tsave[0], y0, None)):
            assert not jnp.iscomplexobj(leaf)


@pytest.mark.run(order=TEST_SHORT)
def test_real_saves_gradient(monkeypatch):
    loss = lambda amplitude: SOLVES['mesolve'](amplitude).expects.real.sum()
    reference = jax.grad(loss)(1.0)
    _force_split_saves(monkeypatch)
    assert jnp.array_equal(jax.grad(loss)(1.0), reference)


@pytest.mark.run(order=TEST_SHORT)
def test_real_saves_after_event(monkeypatch):
    # a solve stopped by an event leaves later saves unfilled (inf): restoring them
    # must not create NaNs
    _force_split_saves(monkeypatch)
    terms = dx.ODETerm(lambda t, y, _: -0.5 * (a.dag() @ a) @ y)  # noqa: ARG005
    event = dx.Event(lambda t, y, *args, **kwargs: y.norm() ** 2 - 0.9)  # noqa: ARG005
    solution = call_diffeqsolve(
        tsave,
        dq.coherent(n, 1.0),
        terms,
        dq.method.Tsit5(),
        None,
        dq.Options(progress_meter=False).initialise(),
        jnp.array([]),
        event=event,
    )
    states = solution.ys[0].to_jax()
    assert jnp.isinf(states[-1]).all()
    assert not jnp.isnan(states).any()
