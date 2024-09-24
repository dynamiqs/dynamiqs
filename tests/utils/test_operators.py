import jax.numpy as jnp

import dynamiqs as dq


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


def test_operators_dispatch():
    dims = (3, 4)
    dim = 20

    assert jnp.allclose(
        dq.eye(*dims, layout=dq.dense).asjaxarray(),
        dq.eye(*dims, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.zero(*dims, layout=dq.dense).asjaxarray(),
        dq.zero(*dims, layout=dq.dia).asjaxarray(),
    )

    # === dq.destroy ===

    assert jnp.allclose(
        dq.destroy(*dims, layout=dq.dense)[0].asjaxarray(),
        dq.destroy(*dims, layout=dq.dia)[0].asjaxarray(),
    )

    assert jnp.allclose(
        dq.destroy(*dims, layout=dq.dense)[1].asjaxarray(),
        dq.destroy(*dims, layout=dq.dia)[1].asjaxarray(),
    )

    # === dq.create ===

    assert jnp.allclose(
        dq.create(*dims, layout=dq.dense)[0].asjaxarray(),
        dq.create(*dims, layout=dq.dia)[0].asjaxarray(),
    )

    assert jnp.allclose(
        dq.create(*dims, layout=dq.dense)[1].asjaxarray(),
        dq.create(*dims, layout=dq.dia)[1].asjaxarray(),
    )

    # === end dq.create ===

    assert jnp.allclose(
        dq.number(dim, layout=dq.dense).asjaxarray(),
        dq.number(dim, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.parity(dim, layout=dq.dense).asjaxarray(),
        dq.parity(dim, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.quadrature(dim, 0.0, layout=dq.dense).asjaxarray(),
        dq.quadrature(dim, 0.0, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.position(dim, layout=dq.dense).asjaxarray(),
        dq.position(dim, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.momentum(dim, layout=dq.dense).asjaxarray(),
        dq.momentum(dim, layout=dq.dia).asjaxarray(),
    )

    assert jnp.allclose(
        dq.sigmax(layout=dq.dense).asjaxarray(), dq.sigmax(layout=dq.dia).asjaxarray()
    )

    assert jnp.allclose(
        dq.sigmay(layout=dq.dense).asjaxarray(), dq.sigmay(layout=dq.dia).asjaxarray()
    )

    assert jnp.allclose(
        dq.sigmaz(layout=dq.dense).asjaxarray(), dq.sigmaz(layout=dq.dia).asjaxarray()
    )

    assert jnp.allclose(
        dq.sigmap(layout=dq.dense).asjaxarray(), dq.sigmap(layout=dq.dia).asjaxarray()
    )

    assert jnp.allclose(
        dq.sigmam(layout=dq.dense).asjaxarray(), dq.sigmam(layout=dq.dia).asjaxarray()
    )
