import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs._utils import cdtype

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_global_dispatch():
    dim = 4

    # default: sparse DIA
    assert dq.eye(dim).layout == dq.dia
    assert dq.eye(dim, layout=dq.dense).layout == dq.dense

    dq.set_layout('dense')
    assert dq.eye(dim).layout == dq.dense
    assert dq.eye(dim, layout=dq.dia).layout == dq.dia

    dq.set_layout('dia')
    assert dq.eye(dim).layout == dq.dia
    assert dq.eye(dim, layout=dq.dense).layout == dq.dense


@pytest.mark.run(order=TEST_INSTANT)
def test_eye():
    # prepare inputs
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.eye, static_argnums=(0,)).trace(*dims_1, layout=dq.dense)
    _jit_static_layout(dq.eye, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dense)
    _jit_static_layout(dq.eye, static_argnums=(0,)).trace(*dims_1, layout=dq.dia)
    _jit_static_layout(dq.eye, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.eye(*dims_2, layout=dq.dense).to_jax(),
        dq.eye(*dims_2, layout=dq.dia).to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_eye_like():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, 2)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.eye_like).trace(x, layout=dq.dense)
    _jit_static_layout(dq.eye_like).trace(x, layout=dq.dia)

    # test default layout
    assert dq.eye_like(dq.sigmax(layout=dq.dense)).layout == dq.dense
    assert dq.eye_like(dq.sigmax(layout=dq.dia)).layout == dq.dia


@pytest.mark.run(order=TEST_INSTANT)
def test_zeros():
    # prepare inputs
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.zeros, static_argnums=(0,)).trace(*dims_1, layout=dq.dense)
    _jit_static_layout(dq.zeros, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dense)
    _jit_static_layout(dq.zeros, static_argnums=(0,)).trace(*dims_1, layout=dq.dia)
    _jit_static_layout(dq.zeros, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.zeros(*dims_2, layout=dq.dense).to_jax(),
        dq.zeros(*dims_2, layout=dq.dia).to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_zeros_like():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, 2)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.zeros_like).trace(x, layout=dq.dense)
    _jit_static_layout(dq.zeros_like).trace(x, layout=dq.dia)

    # test default layout
    assert dq.zeros_like(dq.sigmax(layout=dq.dense)).layout == dq.dense
    assert dq.zeros_like(dq.sigmax(layout=dq.dia)).layout == dq.dia


@pytest.mark.run(order=TEST_INSTANT)
def test_destroy():
    # prepare inputs
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.destroy, static_argnums=(0,)).trace(*dims_1, layout=dq.dense)
    _jit_static_layout(dq.destroy, static_argnums=(0, 1)).trace(
        *dims_2, layout=dq.dense
    )
    _jit_static_layout(dq.destroy, static_argnums=(0,)).trace(*dims_1, layout=dq.dia)
    _jit_static_layout(dq.destroy, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.destroy(*dims_2, layout=dq.dense)[0].to_jax(),
        dq.destroy(*dims_2, layout=dq.dia)[0].to_jax(),
    )

    assert jnp.allclose(
        dq.destroy(*dims_2, layout=dq.dense)[1].to_jax(),
        dq.destroy(*dims_2, layout=dq.dia)[1].to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_create():
    # prepare inputs
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.create, static_argnums=(0,)).trace(*dims_1, layout=dq.dense)
    _jit_static_layout(dq.create, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dense)
    _jit_static_layout(dq.create, static_argnums=(0,)).trace(*dims_1, layout=dq.dia)
    _jit_static_layout(dq.create, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.create(*dims_2, layout=dq.dense)[0].to_jax(),
        dq.create(*dims_2, layout=dq.dia)[0].to_jax(),
    )

    assert jnp.allclose(
        dq.create(*dims_2, layout=dq.dense)[1].to_jax(),
        dq.create(*dims_2, layout=dq.dia)[1].to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_number():
    # prepare inputs
    dim = 4
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.number, static_argnums=(0,)).trace(*dims_1, layout=dq.dense)
    _jit_static_layout(dq.number, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dense)
    _jit_static_layout(dq.number, static_argnums=(0,)).trace(*dims_1, layout=dq.dia)
    _jit_static_layout(dq.number, static_argnums=(0, 1)).trace(*dims_2, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.number(dim, layout=dq.dense).to_jax(), dq.number(dim, layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_parity():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.parity, static_argnums=(0,)).trace(dim, layout=dq.dense)
    _jit_static_layout(dq.parity, static_argnums=(0,)).trace(dim, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.parity(dim, layout=dq.dense).to_jax(), dq.parity(dim, layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_displace():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    jax.jit(dq.displace, static_argnums=(0,)).trace(dim, 0.0)


@pytest.mark.run(order=TEST_INSTANT)
def test_squeeze():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    jax.jit(dq.squeeze, static_argnums=(0,)).trace(dim, 0.0)


@pytest.mark.run(order=TEST_INSTANT)
def test_quadrature():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.quadrature, static_argnums=(0,)).trace(
        dim, 0.0, layout=dq.dense
    )
    _jit_static_layout(dq.quadrature, static_argnums=(0,)).trace(
        dim, 0.0, layout=dq.dia
    )

    # test operators dispatch
    assert jnp.allclose(
        dq.quadrature(dim, 0.0, layout=dq.dense).to_jax(),
        dq.quadrature(dim, 0.0, layout=dq.dia).to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_position():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.position, static_argnums=(0,)).trace(dim, layout=dq.dense)
    _jit_static_layout(dq.position, static_argnums=(0,)).trace(dim, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.position(dim, layout=dq.dense).to_jax(),
        dq.position(dim, layout=dq.dia).to_jax(),
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_momentum():
    # prepare inputs
    dim = 4

    # check that no error is raised while tracing the function
    _jit_static_layout(dq.momentum, static_argnums=(0,)).trace(dim, layout=dq.dense)
    _jit_static_layout(dq.momentum, static_argnums=(0,)).trace(dim, layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.momentum(dim, layout=dq.dense).to_jax(),
        dq.momentum(dim, layout=dq.dia).to_jax(),
    )

# helper
LAYOUTS = [dq.dense, dq.dia]

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_position_norm_convention(layout):
    # prepare inputs
    dim = 4

    x_half = dq.position(dim, layout=layout, norm_convention="half").to_jax()
    x_s2 = dq.position(dim, layout=layout, norm_convention="sqrt2").to_jax()
    x_none = dq.position(dim, layout=layout, norm_convention="none").to_jax()

    # Check expected scaling between conventions
    assert jnp.allclose(x_none, 2.0 * x_half)
    assert jnp.allclose(x_s2, jnp.sqrt(2.0) * x_half)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_momentum_norm_convention(layout):
    # prepare inputs
    dim = 4

    p_half = dq.momentum(dim, layout=layout, norm_convention="half").to_jax()
    p_s2 = dq.momentum(dim, layout=layout, norm_convention="sqrt2").to_jax()
    p_none = dq.momentum(dim, layout=layout, norm_convention="none").to_jax()

    assert jnp.allclose(p_none, 2.0 * p_half)
    assert jnp.allclose(p_s2, jnp.sqrt(2.0) * p_half)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_quadrature_norm_convention(layout):
    # prepare inputs
    dim = 4
    phi = 0.37  # arbitrary nontrivial angle

    q_half = dq.quadrature(dim, phi, layout=layout, norm_convention="half").to_jax()
    q_s2 = dq.quadrature(dim, phi, layout=layout, norm_convention="sqrt2").to_jax()
    q_none = dq.quadrature(dim, phi, layout=layout, norm_convention="none").to_jax()

    assert jnp.allclose(q_none, 2.0 * q_half)
    assert jnp.allclose(q_s2, jnp.sqrt(2.0) * q_half)

# helper
NORMS = ["half", "sqrt2", "none"]

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("norm", NORMS)
def test_quadrature_zero_equals_position_norm_convention(layout, norm):
    # prepare inputs
    dim = 4

    q0 = dq.quadrature(dim, 0.0, layout=layout, norm_convention=norm).to_jax()
    x = dq.position(dim, layout=layout, norm_convention=norm).to_jax()

    assert jnp.allclose(q0, x)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("norm", NORMS)
def test_quadrature_pi_over_2_equals_momentum_norm_convention(layout, norm):
    # prepare inputs
    dim = 4

    q90 = dq.quadrature(dim, jnp.pi / 2, layout=layout, norm_convention=norm).to_jax()
    p = dq.momentum(dim, layout=layout, norm_convention=norm).to_jax()
    assert jnp.allclose(q90, p)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("norm", NORMS)
def test_hermiticity_norm_convention(norm):
    dim = 4
    phi = 0.33

    assert dq.isherm(dq.position(dim, norm_convention=norm))
    assert dq.isherm(dq.momentum(dim, norm_convention=norm))
    assert dq.isherm(dq.quadrature(dim, phi, norm_convention=norm))

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("norm", NORMS)
def test_trace_quadrature_for_each_norm_convention(norm):
    # prepare inputs
    dim = 4

    jax.jit(dq.quadrature, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, 0.1, layout=dq.dense, norm_convention=norm)
    jax.jit(dq.quadrature, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, 0.1, layout=dq.dia, norm_convention=norm)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("norm", NORMS)
def test_trace_position_for_each_norm_convention(norm):
    # prepare inputs
    dim = 4

    jax.jit(dq.position, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, layout=dq.dense, norm_convention=norm)
    jax.jit(dq.position, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, layout=dq.dia, norm_convention=norm)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("norm", NORMS)
def test_trace_momentum_for_each_norm_convention(norm):
    # prepare inputs
    dim = 4

    jax.jit(dq.momentum, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, layout=dq.dense, norm_convention=norm)
    jax.jit(dq.momentum, static_argnums=(0,), static_argnames=("layout", "norm_convention")).trace(dim, layout=dq.dia, norm_convention=norm)

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_defaults_are_half(layout):
    # prepare inputs
    dim = 4

    # Position default
    assert jnp.allclose(
        dq.position(dim, layout=layout).to_jax(),
        dq.position(dim, layout=layout, norm_convention="half").to_jax(),
    )
    # Momentum default
    assert jnp.allclose(
        dq.momentum(dim, layout=layout).to_jax(),
        dq.momentum(dim, layout=layout, norm_convention="half").to_jax(),
    )
    # Quadrature default
    phi = 0.82
    assert jnp.allclose(
        dq.quadrature(dim, phi, layout=layout).to_jax(),
        dq.quadrature(dim, phi, layout=layout, norm_convention="half").to_jax(),
    )

@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize("func_and_args", [
    (dq.position, (4,), {"layout": dq.dense}),
    (dq.momentum, (4,), {"layout": dq.dense}),
    (dq.quadrature, (4, 0.0), {"layout": dq.dense}),
])
def test_invalid_norm_convention_raises_value_error(func_and_args):
    func, args, kwargs = func_and_args
    with pytest.raises(ValueError) as excinfo:
        func(*args, norm_convention="invalid", **kwargs)
    msg = str(excinfo.value)
    assert "Invalid norm_convention=" in msg
    assert "'half'" in msg and "'sqrt2'" in msg and "'none'" in msg

@pytest.mark.run(order=TEST_INSTANT)
def test_sigmax():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.sigmax).trace(layout=dq.dense)
    _jit_static_layout(dq.sigmax).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.sigmax(layout=dq.dense).to_jax(), dq.sigmax(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_sigmay():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.sigmay).trace(layout=dq.dense)
    _jit_static_layout(dq.sigmay).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.sigmay(layout=dq.dense).to_jax(), dq.sigmay(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_sigmaz():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.sigmaz).trace(layout=dq.dense)
    _jit_static_layout(dq.sigmaz).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.sigmaz(layout=dq.dense).to_jax(), dq.sigmaz(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_sigmap():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.sigmap).trace(layout=dq.dense)
    _jit_static_layout(dq.sigmap).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.sigmap(layout=dq.dense).to_jax(), dq.sigmap(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_sigmam():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.sigmam).trace(layout=dq.dense)
    _jit_static_layout(dq.sigmam).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.sigmam(layout=dq.dense).to_jax(), dq.sigmam(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_xyz():
    # check that no error is raised while tracing the function
    _jit_static_layout(dq.xyz).trace(layout=dq.dense)
    _jit_static_layout(dq.xyz).trace(layout=dq.dia)

    # test operators dispatch
    assert jnp.allclose(
        dq.xyz(layout=dq.dense).to_jax(), dq.xyz(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_hadamard():
    # check that no error is raised while tracing the function
    jax.jit(dq.hadamard, static_argnums=(0,)).trace(1)
    jax.jit(dq.hadamard, static_argnums=(0,)).trace(2)

    # test output for:
    # one qubit
    H1 = 2 ** (-1 / 2) * jnp.array([[1, 1], [1, -1]], dtype=cdtype())
    assert jnp.allclose(dq.hadamard(1).to_jax(), H1)

    # two qubits
    H2 = 0.5 * jnp.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=cdtype()
    )
    assert jnp.allclose(dq.hadamard(2).to_jax(), H2)

    # three qubits
    H3 = 2 ** (-3 / 2) * jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=cdtype(),
    )
    assert jnp.allclose(dq.hadamard(3).to_jax(), H3)


@pytest.mark.run(order=TEST_INSTANT)
def test_rx():
    # check that no error is raised while tracing the function
    jax.jit(dq.rx).trace(0.5)


@pytest.mark.run(order=TEST_INSTANT)
def test_ry():
    # check that no error is raised while tracing the function
    jax.jit(dq.ry).trace(0.5)


@pytest.mark.run(order=TEST_INSTANT)
def test_rz():
    # check that no error is raised while tracing the function
    jax.jit(dq.rz).trace(0.5)


@pytest.mark.run(order=TEST_INSTANT)
def test_sgate():
    # check that no error is raised while tracing the function
    jax.jit(dq.sgate).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_tgate():
    # check that no error is raised while tracing the function
    jax.jit(dq.tgate).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_cnot():
    # check that no error is raised while tracing the function
    jax.jit(dq.cnot).trace()


@pytest.mark.run(order=TEST_INSTANT)
def test_toffoli():
    # check that no error is raised while tracing the function
    jax.jit(dq.toffoli).trace()


def _jit_static_layout(f, *args, **kwargs):
    return jax.jit(f, *args, **kwargs, static_argnames=('layout',))
