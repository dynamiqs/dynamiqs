import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Forward
from dynamiqs.method import TimeEvolution, Tsit5

from ..order import TEST_INSTANT, TEST_LONG
from ..systems import dense_driven_ocavity, dia_driven_ocavity


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('system', 'vectorized'),
    [
        (dense_driven_ocavity, False),
        (dia_driven_ocavity, False),
        (dense_driven_ocavity, True),
    ],
    ids=['dense', 'dia', 'dense-vectorized'],
)
def test_correctness(system, vectorized):
    result = system.run(TimeEvolution(), vectorized=vectorized)
    assert result.infos.converged
    error = jnp.linalg.norm(result.state.to_jax() - system.steady_state().to_jax())
    assert error <= 1e-3
    assert jnp.allclose(result.expects, system.steady_expect(), rtol=1e-3, atol=1e-4)


@pytest.mark.run(order=TEST_LONG)
def test_gradient():
    # the gradient error is 10-100 times the state error, so tighten the defaults
    system = dense_driven_ocavity
    method = TimeEvolution(ode_method=Tsit5(rtol=1e-10, atol=1e-10), rtol=1e-7)

    def loss(params):
        result = system.run(method, gradient=Forward(), params=params)
        return system.loss_state(result.state)

    grads = jax.jacfwd(loss)(system.params_default)
    true_grads = system.steady_grads_state()
    assert jnp.allclose(jnp.array(grads), jnp.array(true_grads), rtol=1e-3, atol=1e-4)


@pytest.mark.run(order=TEST_LONG)
def test_reverse_mode_without_gradient_raises():
    # control of `test_reverse_mode_gradient_unsupported`: without a gradient, the
    # time evolution must not fall back to reverse mode
    system = dense_driven_ocavity

    def loss(params):
        return system.loss_state(system.run(TimeEvolution(), params=params).state)

    with pytest.raises(ValueError, match='Reverse-mode differentiation'):
        jax.grad(loss)(system.params_default)


@pytest.mark.run(order=TEST_LONG)
def test_not_converged():
    # control of `test_correctness`: too few steps to reach the steady state
    max_steps = 5
    method = TimeEvolution(ode_method=Tsit5(max_steps=max_steps))
    result = dense_driven_ocavity.run(method)
    assert not result.infos.converged
    assert result.infos.steps.nsteps == max_steps


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('rho0', 'expected'),
    [(None, dq.eye(2) / 2), (dq.basis(2, 0), dq.basis_dm(2, 0))],
    ids=['default', 'ket'],
)
def test_rho0_selects_steady_state(rho0, expected):
    # every diagonal state is steady under pure dephasing
    result = dq.steadystate(dq.zeros(2), [dq.sigmaz()], method=TimeEvolution(rho0))
    assert jnp.allclose(result.state.to_jax(), expected.to_jax(), atol=1e-5)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('H', 'jump_ops', 'error', 'match'),
    [
        (
            dq.modulated(jnp.cos, dq.sigmax()),
            [dq.sigmam()],
            TypeError,
            'must be time-independent',
        ),
        (
            dq.constant(dq.sigmax()).clip(0.0, 1.0),
            [dq.sigmam()],
            TypeError,
            'must be time-independent',
        ),
        (dq.sigmax(), [], ValueError, 'at least one jump operator'),
    ],
    ids=['time-dependent', 'clipped-constant', 'no-jump-ops'],
)
def test_invalid_args(H, jump_ops, error, match):
    with pytest.raises(error, match=match):
        dq.steadystate(H, jump_ops)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    'gradient',
    [dq.gradient.BackwardCheckpointed(), dq.gradient.Direct()],
    ids=['checkpointed', 'direct'],
)
def test_reverse_mode_gradient_unsupported(gradient):
    a = dq.destroy(4)
    with pytest.raises(ValueError, match='does not support gradient'):
        dq.steadystate(a.dag() @ a, [a], gradient=gradient)
