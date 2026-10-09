import diffrax as dx
import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.integrators.core.unrolled_solvers import UnrolledDopri8, UnrolledTsit5

from ..order import TEST_LONG


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


@pytest.mark.run(order=TEST_LONG)
def test_unrolled_stages_in_mesolve(monkeypatch):
    # through `dq.mesolve`: the stages are unrolled on GPU only, so CI forces them here,
    # against the CPU loop
    reference, reference_steps = _solve()(1.3)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    jax.clear_caches()
    value, steps = _solve()(1.3)
    assert steps == reference_steps
    assert jnp.allclose(value, reference, rtol=1e-5)
    grad = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0])(1.3)
    jax.clear_caches()
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    grad_ref = jax.jacfwd(lambda x: _solve(dq.gradient.Forward())(x)[0])(1.3)
    assert jnp.allclose(grad, grad_ref, rtol=1e-4)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('diffrax_solver', 'solver'),
    [(dx.Tsit5, UnrolledTsit5), (dx.Dopri8, UnrolledDopri8)],
)
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.filterwarnings('ignore:Complex dtype support in Diffrax:UserWarning')
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

    # in double precision: in single precision, round-off moves the steps
    with jax.enable_x64(True):
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
