from functools import partial

import jax
import jax.numpy as jnp
import pytest

from dynamiqs.gradient import Implicit
from dynamiqs.method import LyapunovGMRES

from ..order import TEST_LONG
from ..systems import dense_thermal_cavity, dia_thermal_cavity


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('system', [dense_thermal_cavity, dia_thermal_cavity])
def test_correctness(system):
    result = system.run(LyapunovGMRES())

    assert result.infos.converged
    assert jnp.allclose(result.state.to_jax(), system.state().to_jax(), atol=1e-3)
    assert jnp.allclose(result.expects.real, system.expect(), rtol=1e-3, atol=1e-4)


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize(
    ('system', 'jax_jac'),
    [
        (dense_thermal_cavity, jax.jacrev),
        (dia_thermal_cavity, jax.jacrev),
        (dense_thermal_cavity, jax.jacfwd),
    ],
    ids=['dense-rev', 'dia-rev', 'dense-fwd'],
)
def test_gradient(system, jax_jac):
    def expects(params):
        method = LyapunovGMRES()
        return system.run(method, gradient=Implicit(), params=params).expects.real

    grads = jax_jac(expects)(system.params_default)

    allclose = partial(jnp.allclose, rtol=1e-3, atol=1e-4)
    assert all(jax.tree.leaves(jax.tree.map(allclose, grads, system.grads_expect())))


@pytest.mark.run(order=TEST_LONG)
def test_not_converged():
    max_iter = 1
    method = LyapunovGMRES(krylov_size=2, recycling=0, max_iter=max_iter)

    result = dense_thermal_cavity.run(method)

    assert not result.infos.converged
    assert result.infos.niter == max_iter
