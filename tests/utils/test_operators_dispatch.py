import jax.numpy as jnp

import dynamiqs as dq


def test_operators_dispatch():
    dims = (3, 4)

    assert jnp.allclose(
        dq.eye(*dims, matrix_format=dq.dense).to_jax(),
        dq.eye(*dims, matrix_format=dq.dia).to_jax(),
    )
