import diffrax as dx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core import unrolled_solvers
from dynamiqs.integrators.core.unrolled_solvers import _UnrolledERK

from ..order import TEST_INSTANT, TEST_LONG, TEST_SHORT


def _solve(gradient=None):
    # the population of a driven, damped cavity with a pulse edge (a jump time) at the
    # save times, as a function of the drive amplitude
    a = dq.destroy(8)
    tsave = jnp.linspace(0.0, 2.0, 11)

    def population(amplitude):
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
        return result.expects[0].real, result.infos.nsteps

    return population


def _derivatives():
    # forward and reverse mode derivatives of the final population of `_solve`
    jvp = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0][-1])(1.3)
    vjp = jax.grad(lambda x: _solve(dq.gradient.BackwardCheckpointed())(x)[0][-1])(1.3)
    return jvp, vjp


@pytest.fixture(scope='module')
def diffrax_reference():
    # `_solve` and its derivatives with Diffrax's steps, computed once
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
        monkeypatch.setattr(_UnrolledERK, '_supported', lambda *args: False)  # noqa: ARG005
        jax.clear_caches()
        reference = (*_solve()(1.3), *_derivatives())
    jax.clear_caches()
    return reference


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('backend', 'odd'),
    [('gpu', False), ('gpu', True), ('cpu', False)],
    ids=['unrolled', 'unrolled-odd', 'looped'],
)
def test_unrolled_matches_diffrax(monkeypatch, diffrax_reference, backend, odd):
    # the stages are unrolled on GPU only, so CI forces them here, and looped over on
    # CPU; odd: with the stages of large states written out with an odd size (forced
    # on a small state)
    value_ref, steps_ref, jvp_ref, vjp_ref = diffrax_reference
    monkeypatch.setattr(jax, 'default_backend', lambda: backend)
    if odd:
        monkeypatch.setattr(unrolled_solvers, '_MIN_SIZE_ODD_STAGES', 0)
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
@pytest.mark.parametrize('nodes', [False, True], ids=['stages', 'nodes'])
def test_unrolled_interpolation(unrolled, diffrax, nodes):
    # the interpolation of the stages kept apart (without the inputs whose weight is
    # always zero), or of its values at fixed nodes (large states), is Diffrax's
    # interpolation of the stacked stages
    nk = len(diffrax.tableau.c) + 1
    keys = jax.random.split(jax.random.key(0), nk + 2)
    y0, y1, *ks = [dq.random.complex(key, (3, 2)) for key in keys]
    times = dict(t0=0.2, t1=0.7)
    if nodes:
        degree = unrolled._interpolation_degree
        stages = unrolled_solvers._nodes(unrolled.interpolation_cls, degree, y0, y1, ks)
    else:
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
@pytest.mark.parametrize('nodes', [False, True], ids=['stages', 'nodes'])
def test_unrolled_dense_output(monkeypatch, nodes):
    # the unrolled steps keep their stages apart for the interpolation at the save
    # times and in the root finder, or its values at fixed nodes (large states, forced
    # on a small one): the saved values and click times must be Diffrax's
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    monkeypatch.setattr(_UnrolledERK, '_supported', lambda *args: False)  # noqa: ARG005
    jax.clear_caches()
    expects, steps, states, clicktimes = _dense_output()
    monkeypatch.undo()
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    # whether a step hands the save loop the values at nodes (seen when it is traced)
    calls = []
    nodes_of = unrolled_solvers._nodes

    def spy(*args):
        calls.append(None)
        return nodes_of(*args)

    monkeypatch.setattr(unrolled_solvers, '_nodes', spy)
    if nodes:
        monkeypatch.setattr(unrolled_solvers, '_MIN_SIZE_ODD_STAGES', 0)
    jax.clear_caches()
    expects_unrolled, steps_unrolled, states_unrolled, clicktimes_unrolled = (
        _dense_output()
    )
    jax.clear_caches()
    assert bool(calls) == nodes
    assert steps_unrolled == steps
    assert steps < expects.shape[-1]  # several save times per step
    assert jnp.isfinite(clicktimes).any()  # clicks found by the root finder
    assert jnp.allclose(expects_unrolled, expects, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(states_unrolled, states, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(clicktimes_unrolled, clicktimes, rtol=1e-10, equal_nan=True)


# double precision, for a tight comparison: XLA fuses the flattened sums in a
# different order
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.usefixtures('_double_precision')
def test_odd_sized_stages_match(monkeypatch):
    # large states write their stages out with an odd number of elements: forced here
    # on a small one, the solve under nested vmaps and its steps are unchanged (its
    # derivatives are compared with Diffrax's in `test_unrolled_matches_diffrax`)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')

    def run():
        jax.clear_caches()
        amplitudes = jnp.array([[1.3, 0.4], [0.9, 1.1]])
        return jax.vmap(jax.vmap(_solve()))(amplitudes)

    values, steps = run()
    monkeypatch.setattr(unrolled_solvers, '_MIN_SIZE_ODD_STAGES', 0)
    values_odd, steps_odd = run()
    assert jnp.array_equal(steps_odd, steps)
    assert jnp.allclose(values_odd, values, rtol=1e-12)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('diffrax_solver', 'solver'),
    [
        (dx.Tsit5, unrolled_solvers.UnrolledTsit5),
        (dx.Dopri8, unrolled_solvers.UnrolledDopri8),
    ],
)
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.filterwarnings('ignore:Complex dtype support in Diffrax:UserWarning')
@pytest.mark.usefixtures('_double_precision')
def test_steps_match_diffrax(monkeypatch, diffrax_solver, solver, backend):
    # dynamiqs' solvers loop over the stages on CPU and unroll them on GPU: both take
    # Diffrax's steps, here on a complex ODE with a jump time (where the first stage
    # is evaluated again), and its derivative
    generator = jnp.array([[0.0, 1.0], [1.0, 0.5]]) - 0.1j * jnp.eye(2)

    def solve(solver, amplitude):
        def vf(t, y, _):
            return -1j * amplitude * jnp.cos(3.0 * t) * (generator @ y)

        controller = dx.PIDController(rtol=1e-8, atol=1e-8, jump_ts=jnp.array([0.7]))
        solution = dx.diffeqsolve(
            dx.ODETerm(vf),
            solver,
            t0=0.0,
            t1=2.0,
            dt0=0.01,
            y0=jnp.array([1.0, 0.0], dtype=jnp.complex128),
            adjoint=dx.ForwardMode(),
            saveat=dx.SaveAt(ts=jnp.linspace(0.0, 2.0, 11)),
            stepsize_controller=controller,
        )
        return solution.ys, solution.stats['num_steps']

    def derivative(solver):
        return jax.jacfwd(lambda x: solve(solver, x)[0].real)(1.3)

    reference, reference_steps = solve(diffrax_solver(), 1.3)
    reference_derivative = derivative(diffrax_solver())
    monkeypatch.setattr(jax, 'default_backend', lambda: backend)
    jax.clear_caches()
    ys, steps = solve(solver(), 1.3)
    assert steps == reference_steps
    assert jnp.allclose(ys, reference, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        derivative(solver()), reference_derivative, rtol=1e-10, atol=1e-10
    )


@pytest.mark.run(order=TEST_SHORT)
def test_unrolled_stages_reuse_the_vector_field_trace(monkeypatch):
    # the vector field is jitted: tracing a solve calls the Python of H(t) a few
    # times, not once per Runge-Kutta stage and per trace of the step
    calls = []

    def solve():
        a = dq.destroy(8)

        def coefficient(t):
            calls.append(t)
            return jnp.where(t < 0.7, 1.0, 0.3)

        H = a.dag() @ a + dq.modulated(coefficient, a + a.dag(), discontinuity_ts=[0.7])
        tsave = jnp.linspace(0.0, 2.0, 11)
        return dq.mesolve(H, [0.5 * a], dq.fock(8, 0), tsave, progress_meter=False)

    counts = {}
    for backend in ('cpu', 'gpu'):
        monkeypatch.setattr(jax, 'default_backend', lambda b=backend: b)
        jax.clear_caches()
        calls.clear()
        jax.jit(solve).trace()
        counts[backend] = len(calls)
    jax.clear_caches()
    # the unrolled stages (gpu) trace it about as often as the stage loop (cpu)
    assert counts['gpu'] <= counts['cpu'] + 4
