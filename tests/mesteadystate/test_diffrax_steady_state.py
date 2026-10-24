from functools import partial

import jax
import jax.numpy as jnp
import pytest

from dynamiqs.gradient import BackwardCheckpointed, Implicit
from dynamiqs.method import DiffraxSteadyState, Expm, Kvaerno5, Tsit5

from ..order import TEST_INSTANT, TEST_LONG
from ..systems import dense_thermal_cavity, dia_thermal_cavity

# few steps, so that a stop that never fires fails fast
MAX_STEPS = 1000
METHOD = DiffraxSteadyState(Kvaerno5(max_steps=MAX_STEPS))
# differentiating through the time integration is ~10x less accurate than the
# implicit gradient at the same stopping tolerance
TIGHT_METHOD = DiffraxSteadyState(Kvaerno5(max_steps=MAX_STEPS), rtol=1e-6, atol=1e-6)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('system', 'me_method'),
    [
        (dense_thermal_cavity, Kvaerno5(max_steps=MAX_STEPS)),
        (dia_thermal_cavity, Kvaerno5(max_steps=MAX_STEPS)),
        (dense_thermal_cavity, Tsit5(max_steps=MAX_STEPS)),
    ],
    ids=['dense-kvaerno5', 'dia-kvaerno5', 'dense-tsit5'],
)
def test_correctness(system, me_method):
    result = system.run(DiffraxSteadyState(me_method))

    assert result.infos.converged
    assert jnp.allclose(result.state.to_jax(), system.state().to_jax(), atol=1e-3)
    assert jnp.allclose(result.expects.real, system.expect(), rtol=1e-3, atol=1e-4)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('system', 'method', 'gradient', 'jax_jac'),
    [
        (dense_thermal_cavity, METHOD, Implicit(), jax.jacrev),
        (dia_thermal_cavity, METHOD, Implicit(), jax.jacrev),
        (dense_thermal_cavity, METHOD, Implicit(), jax.jacfwd),
        (dense_thermal_cavity, TIGHT_METHOD, BackwardCheckpointed(), jax.jacrev),
    ],
    ids=[
        'dense-implicit-rev',
        'dia-implicit-rev',
        'dense-implicit-fwd',
        'dense-checkpointed-rev',
    ],
)
def test_gradient(system, method, gradient, jax_jac):
    def expects(params):
        return system.run(method, gradient=gradient, params=params).expects.real

    grads = jax_jac(expects)(system.params_default)

    allclose = partial(jnp.allclose, rtol=1e-3, atol=1e-4)
    assert all(jax.tree.leaves(jax.tree.map(allclose, grads, system.grads_expect())))


@pytest.mark.run(order=TEST_LONG)
def test_not_converged():
    max_steps = 3
    method = DiffraxSteadyState(Kvaerno5(max_steps=max_steps))

    result = dense_thermal_cavity.run(method)

    assert not result.infos.converged
    assert result.infos.steps.nsteps == max_steps


@pytest.mark.run(order=TEST_INSTANT)
def test_unsupported_me_method():
    with pytest.raises(TypeError, match='Argument `me_method` must be one of'):
        DiffraxSteadyState(Expm())
