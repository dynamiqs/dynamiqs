import jax.numpy as jnp
import pytest

import dynamiqs as dq

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
