import diffrax as dx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core import unrolled_solvers

from ..order import TEST_INSTANT, TEST_LONG


def _solve(gradient=None):
    # a driven, damped cavity with a pulse edge (a jump time), and its derivative
    # with respect to the drive amplitude
    a = dq.destroy(8)
    tsave = jnp.linspace(0.0, 2.0, 11)

    def final_population(amplitude):
        H = a.dag() @ a + dq.modulated(
            lambda t: amplitude * jnp.where(t < 0.7, 1.0, 0.3),
            a + a.dag(),
            discontinuity_ts=[0.7],
        )
        result = dq.mesolve(
            H,
            [0.5 * a],
            dq.fock(8, 0),
            tsave,
            exp_ops=[a.dag() @ a],
            gradient=gradient,
            progress_meter=False,
        )
        return result.expects[0, -1].real, result.infos.nsteps

    return final_population


def _derivatives():
    # forward and reverse mode derivatives of `_solve`
    jvp = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0])(1.3)
    vjp = jax.grad(lambda x: _solve(dq.gradient.BackwardCheckpointed())(x)[0])(1.3)
    return jvp, vjp


@pytest.fixture(scope='module')
def diffrax_reference():
    # `_solve` and its derivatives with Diffrax's steps, computed once
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
        jax.clear_caches()
        reference = (*_solve()(1.3), *_derivatives())
    jax.clear_caches()
    return reference


@pytest.mark.run(order=TEST_LONG)
def test_unrolled_matches_diffrax(monkeypatch, diffrax_reference):
    # the stages are unrolled on GPU only, so CI forces them here
    value_ref, steps_ref, jvp_ref, vjp_ref = diffrax_reference
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    jax.clear_caches()
    value, steps = _solve()(1.3)
    jvp, vjp = _derivatives()
    jax.clear_caches()
    assert steps == steps_ref
    assert jnp.allclose(value, value_ref, rtol=1e-5)
    assert jnp.allclose(jvp, jvp_ref, rtol=1e-4)
    assert jnp.allclose(vjp, vjp_ref, rtol=1e-4)


@pytest.fixture
def _double_precision():
    # so that the unrolled and Diffrax steps are the same
    previous = jax.config.read('jax_enable_x64')
    dq.set_precision('double')
    yield
    dq.set_precision('double' if previous else 'single')


def _dense_output():
    # a driven, damped cavity with a pulse edge, saved at more times than it takes
    # steps, and a jump trajectory whose clicks are found by root finding (the
    # interpolation is evaluated inside the root finder)
    a = dq.destroy(8)
    tsave = jnp.linspace(0.0, 2.0, 301)
    drive = lambda t: 1.3 * jnp.where(t < 0.7, 1.0, 0.3)
    H = a.dag() @ a + dq.modulated(drive, a + a.dag(), discontinuity_ts=[0.7])
    me = dq.mesolve(
        H,
        [0.5 * a],
        dq.fock(8, 0),
        tsave,
        exp_ops=[a],
        method=dq.method.Tsit5(rtol=1e-8, atol=1e-8),
        progress_meter=False,
    )
    root_finder = optx.Newton(1e-8, 1e-8, jtu.Partial(optx.rms_norm))
    event = dq.method.Event(
        noclick_method=dq.method.Tsit5(rtol=1e-8, atol=1e-8), root_finder=root_finder
    )
    keys = jax.random.split(jax.random.key(42), 2)
    jsse = dq.jssesolve(H, [a], dq.coherent(8, 1.0), tsave, keys, method=event)
    return me.expects, me.infos.nsteps, jsse.states.to_jax(), jsse.clicktimes


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('unrolled', 'diffrax'),
    [
        (unrolled_solvers.UnrolledTsit5, dx.Tsit5),
        (unrolled_solvers.UnrolledDopri5, dx.Dopri5),
        (unrolled_solvers.UnrolledDopri8, dx.Dopri8),
    ],
)
def test_unrolled_interpolation(unrolled, diffrax):
    # the interpolation of the stages kept apart (without the inputs whose weight is
    # always zero) is Diffrax's interpolation of the stacked stages
    nk = len(diffrax.tableau.c) + 1
    keys = jax.random.split(jax.random.key(0), nk + 2)
    y0, y1, *ks = [dq.random.complex(key, (3, 2)) for key in keys]
    times = dict(t0=0.2, t1=0.7)
    stages = unrolled_solvers._Stages(tuple(ks))
    interpolation = unrolled.interpolation_cls(**times, y0=y0, y1=y1, k=stages)
    reference = diffrax.interpolation_cls(**times, y0=y0, y1=y1, k=jnp.stack(ks))
    for t in (0.2, 0.33, 0.5, 0.7):
        value = interpolation.evaluate(t)
        assert jnp.allclose(value, reference.evaluate(t), rtol=1e-5, atol=1e-6)
    increment = interpolation.evaluate(0.3, 0.6)
    assert jnp.allclose(increment, reference.evaluate(0.3, 0.6), rtol=1e-5, atol=1e-6)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.usefixtures('_double_precision')
def test_unrolled_dense_output(monkeypatch):
    # the unrolled steps keep their stages apart for the interpolation at the save
    # times and in the root finder: the saved values and click times must be Diffrax's
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    jax.clear_caches()
    expects, steps, states, clicktimes = _dense_output()
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    jax.clear_caches()
    expects_unrolled, steps_unrolled, states_unrolled, clicktimes_unrolled = (
        _dense_output()
    )
    jax.clear_caches()
    assert steps_unrolled == steps
    assert steps < expects.shape[-1]  # several save times per step
    assert jnp.isfinite(clicktimes).any()  # clicks found by the root finder
    assert jnp.allclose(expects_unrolled, expects, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(states_unrolled, states, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(clicktimes_unrolled, clicktimes, rtol=1e-10, equal_nan=True)
