from functools import partial

import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.gradient import Implicit
from dynamiqs.method import LyapunovGMRES

from ..order import TEST_INSTANT, TEST_LONG
from ..systems import dense_thermal_cavity, dia_thermal_cavity
from ..systems.steady_state_system import DrivenThermalCavity


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('system', [dense_thermal_cavity, dia_thermal_cavity])
def test_correctness(system):
    result = system.run(LyapunovGMRES())

    assert result.infos.converged
    assert jnp.allclose(result.state.to_jax(), system.state().to_jax(), atol=1e-3)
    assert jnp.allclose(result.expects.real, system.expect(), rtol=1e-3, atol=1e-4)


@pytest.mark.run(order=TEST_LONG)
def test_correctness_singular_deflation():
    # Tr[S^-1(I)] = -1: singular preconditioner for the deflation L + |I><I|
    kappa = 7.1044368
    system = DrivenThermalCavity(
        n=16, delta=0.7, eps=0.4, kappa=kappa, nth=0.2, layout=dq.dense
    )

    result = system.run(LyapunovGMRES())

    assert result.infos.converged
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


@pytest.mark.run(order=TEST_LONG)
def test_not_converged_dark_state():
    # the ground state is dark and an eigenstate of H
    H = dq.sigmaz()
    Ls = [dq.sigmam()]

    result = dq.mesteadystate(H, Ls, method=LyapunovGMRES())

    assert not result.infos.converged


@pytest.mark.run(order=TEST_LONG)
def test_gradient_rho0():
    # rho0 is only an initial guess
    system = dense_thermal_cavity
    params = system.params_default
    alpha = 0.5

    def expects(alpha):
        rho0 = dq.coherent_dm(system.n, alpha)
        H, Ls, Es = system.H(params), system.Ls(params), system.Es(params)
        method = LyapunovGMRES()
        result = dq.mesteadystate(H, Ls, rho0=rho0, exp_ops=Es, method=method)
        return result.expects.real

    grads = jax.jacrev(expects)(alpha)

    assert jnp.allclose(grads, 0)


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'krylov_size': 1}, 'Argument `krylov_size` must be an integer >= 2'),
        ({'recycling': -1}, 'Argument `recycling` must be an integer >= 0'),
        ({'max_iter': 0}, 'Argument `max_iter` must be an integer >= 1'),
    ],
    ids=['krylov_size', 'recycling', 'max_iter'],
)
def test_invalid_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LyapunovGMRES(**kwargs)
