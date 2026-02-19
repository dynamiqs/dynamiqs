import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT, TEST_SHORT


@pytest.mark.run(order=TEST_SHORT)
def test_wigner_cat():
    # parameters
    n = 16
    alpha = 2.0
    npixels = 101
    xmax, ymax = 5, 3

    # state to test
    cat = dq.unit(dq.coherent(n, alpha) + dq.coherent(n, -alpha))

    # reference expectation value
    nbar = dq.expect(dq.number(n), cat).real

    # compute wigner and expectation value
    x_vec, y_vec, wig = dq.wigner(cat, npixels=npixels, xmax=xmax, ymax=ymax)
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    x_vec, y_vec = jnp.meshgrid(x_vec, y_vec, indexing='ij')
    nbar_vec = x_vec**2 + y_vec**2 - 0.5
    nbar_wig = jnp.sum(wig * nbar_vec) * dx * dy

    assert jnp.allclose(nbar, nbar_wig)


@pytest.mark.run(order=TEST_SHORT)
def test_wigner_coherent():
    # parameters
    n = 16
    alpha = 2.0
    npixels = 101
    xmax, ymax = 5, 5

    # state to test
    coh = dq.coherent(n, alpha)

    # reference expectation value
    a = dq.expect(dq.destroy(n), coh)

    # compute wigner and expectation value
    x_vec, y_vec, wig = dq.wigner(coh, npixels=npixels, xmax=xmax, ymax=ymax)
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    x_vec, y_vec = jnp.meshgrid(x_vec, y_vec, indexing='ij')
    a_vec = x_vec + 1j * y_vec
    a_wig = jnp.sum(wig * a_vec) * dx * dy

    assert jnp.allclose(a, a_wig)


@pytest.mark.run(order=TEST_INSTANT)
def test_wigner_tracing():
    # prepare inputs
    state = dq.coherent(8, 1.0)
    xvec = jnp.linspace(-3, 3, 101)
    yvec = jnp.linspace(-2, 2, 51)

    # check that no error is raised while tracing the function
    jax.jit(dq.wigner).trace(state)
    jax.jit(dq.wigner).trace(state, xvec=xvec)
    jax.jit(dq.wigner).trace(state, yvec=yvec)
    jax.jit(dq.wigner).trace(state, xvec=xvec, yvec=yvec)

NORMS_WIGNER = ["half", "sqrt2", "none"]

@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize("norm", NORMS_WIGNER)
def test_wigner_utils_valid_norm_convention(norm):
    psi = dq.coherent(10, 2.0)
    xvec, yvec, w = dq.wigner(psi, norm_convention=norm)
    assert w.shape == (201, 201)

@pytest.mark.run(order=TEST_SHORT)
def test_wigner_utils_invalid_norm_convention():
    psi = dq.coherent(10, 2.0)
    with pytest.raises(ValueError) as excinfo:
        dq.wigner(psi, norm_convention="invalid")
    msg = str(excinfo.value)
    assert "Invalid norm_convention=" in msg
    assert "'half'" in msg and "'sqrt2'" in msg and "'none'" in msg

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("norm", NORMS_WIGNER)
def test_wigner_tracing_norm_convention(norm):
    state = dq.coherent(8, 1.0)
    jax.jit(dq.wigner, static_argnames=("norm_convention",)).trace(state, norm_convention=norm)