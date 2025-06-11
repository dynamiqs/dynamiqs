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
    assert isinstance(dq.eye(dim), dq.SparseDIAQArray)
    assert isinstance(dq.eye(dim, layout=dq.dense), dq.DenseQArray)

    dq.set_layout('dense')
    assert isinstance(dq.eye(dim), dq.DenseQArray)
    assert isinstance(dq.eye(dim, layout=dq.dia), dq.SparseDIAQArray)

    dq.set_layout('dia')
    assert isinstance(dq.eye(dim), dq.SparseDIAQArray)
    assert isinstance(dq.eye(dim, layout=dq.dense), dq.DenseQArray)


@pytest.mark.run(order=TEST_INSTANT)
def test_operators_dispatch():
    dims = (3, 4)
    dim = 20

    assert jnp.allclose(
        dq.eye(*dims, layout=dq.dense).to_jax(), dq.eye(*dims, layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.zeros(*dims, layout=dq.dense).to_jax(),
        dq.zeros(*dims, layout=dq.dia).to_jax(),
    )

    # === dq.destroy ===

    assert jnp.allclose(
        dq.destroy(*dims, layout=dq.dense)[0].to_jax(),
        dq.destroy(*dims, layout=dq.dia)[0].to_jax(),
    )

    assert jnp.allclose(
        dq.destroy(*dims, layout=dq.dense)[1].to_jax(),
        dq.destroy(*dims, layout=dq.dia)[1].to_jax(),
    )

    # === dq.create ===

    assert jnp.allclose(
        dq.create(*dims, layout=dq.dense)[0].to_jax(),
        dq.create(*dims, layout=dq.dia)[0].to_jax(),
    )

    assert jnp.allclose(
        dq.create(*dims, layout=dq.dense)[1].to_jax(),
        dq.create(*dims, layout=dq.dia)[1].to_jax(),
    )

    # === end dq.create ===

    assert jnp.allclose(
        dq.number(dim, layout=dq.dense).to_jax(), dq.number(dim, layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.parity(dim, layout=dq.dense).to_jax(), dq.parity(dim, layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.quadrature(dim, 0.0, layout=dq.dense).to_jax(),
        dq.quadrature(dim, 0.0, layout=dq.dia).to_jax(),
    )

    assert jnp.allclose(
        dq.position(dim, layout=dq.dense).to_jax(),
        dq.position(dim, layout=dq.dia).to_jax(),
    )

    assert jnp.allclose(
        dq.momentum(dim, layout=dq.dense).to_jax(),
        dq.momentum(dim, layout=dq.dia).to_jax(),
    )

    assert jnp.allclose(
        dq.xyz(layout=dq.dense).to_jax(), dq.xyz(layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.sigmax(layout=dq.dense).to_jax(), dq.sigmax(layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.sigmay(layout=dq.dense).to_jax(), dq.sigmay(layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.sigmaz(layout=dq.dense).to_jax(), dq.sigmaz(layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.sigmap(layout=dq.dense).to_jax(), dq.sigmap(layout=dq.dia).to_jax()
    )

    assert jnp.allclose(
        dq.sigmam(layout=dq.dense).to_jax(), dq.sigmam(layout=dq.dia).to_jax()
    )


@pytest.mark.run(order=TEST_INSTANT)
def test_hadamard():
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


@pytest.mark.parametrize('layout', [None, dq.dense, dq.dia])
@pytest.mark.run(order=TEST_INSTANT)
def test_tracing(layout):
    # prepare random keys and dimensions
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (2, 2))
    dim = 2
    dims_1 = (2,)
    dims_2 = (2, 3)

    # check that no error is raised while tracing the functions
    jit_static_layout(dq.eye, static_argnums=(0,)).trace(*dims_1, layout=layout)
    jit_static_layout(dq.eye, static_argnums=(0, 1)).trace(*dims_2, layout=layout)
    jit_static_layout(dq.eye_like).trace(x, layout=layout)
    jit_static_layout(dq.zeros, static_argnums=(0,)).trace(*dims_1, layout=layout)
    jit_static_layout(dq.zeros, static_argnums=(0, 1)).trace(*dims_2, layout=layout)
    jit_static_layout(dq.zeros_like).trace(x, layout=layout)
    jit_static_layout(dq.destroy, static_argnums=(0,)).trace(*dims_1, layout=layout)
    jit_static_layout(dq.destroy, static_argnums=(0, 1)).trace(*dims_2, layout=layout)
    jit_static_layout(dq.create, static_argnums=(0,)).trace(*dims_1, layout=layout)
    jit_static_layout(dq.create, static_argnums=(0, 1)).trace(*dims_2, layout=layout)
    jit_static_layout(dq.number, static_argnums=(0,)).trace(*dims_1, layout=layout)
    jit_static_layout(dq.number, static_argnums=(0, 1)).trace(*dims_2, layout=layout)
    jit_static_layout(dq.parity, static_argnums=(0,)).trace(dim, layout=layout)
    jax.jit(dq.displace, static_argnums=(0,)).trace(dim, 0.0)
    jax.jit(dq.squeeze, static_argnums=(0,)).trace(dim, 0.0)
    jit_static_layout(dq.quadrature, static_argnums=(0,)).trace(dim, 0.0, layout=layout)
    jit_static_layout(dq.position, static_argnums=(0,)).trace(dim, layout=layout)
    jit_static_layout(dq.momentum, static_argnums=(0,)).trace(dim, layout=layout)
    jit_static_layout(dq.xyz).trace(layout=layout)
    jit_static_layout(dq.sigmax).trace(layout=layout)
    jit_static_layout(dq.sigmay).trace(layout=layout)
    jit_static_layout(dq.sigmaz).trace(layout=layout)
    jit_static_layout(dq.sigmap).trace(layout=layout)
    jit_static_layout(dq.sigmam).trace(layout=layout)
    jax.jit(dq.hadamard, static_argnums=(0,)).trace(1)
    jax.jit(dq.hadamard, static_argnums=(0,)).trace(2)
    jax.jit(dq.rx).trace(0.5)
    jax.jit(dq.ry).trace(0.5)
    jax.jit(dq.rz).trace(0.5)
    jax.jit(dq.sgate).trace()
    jax.jit(dq.tgate).trace()
    jax.jit(dq.cnot).trace()
    jax.jit(dq.toffoli).trace()


def jit_static_layout(f, *args, **kwargs):
    return jax.jit(f, *args, **kwargs, static_argnames=('layout',))
