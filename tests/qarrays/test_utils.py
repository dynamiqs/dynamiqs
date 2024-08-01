import jax.numpy as jnp
import jax.random as jr

import dynamiqs as dq


def test_stack_simple(rtol=1e-05, atol=1e-08):
    n = 3
    key = jr.key(42)

    jax_m = jr.uniform(key, (n, n))
    dense_m = dq.asqarray(jax_m)
    sparse_m = dq.to_sparse_dia(jax_m)

    print(sparse_m.diags)

    assert jnp.allclose(
        dq.stack([dense_m, 2 * dense_m]).to_jax(),
        jnp.stack([jax_m, 2 * jax_m]),
        rtol=rtol,
        atol=atol,
    )

    assert jnp.allclose(
        dq.stack([sparse_m, 2 * sparse_m]).to_jax(),
        jnp.stack([jax_m, 2 * jax_m]),
        rtol=rtol,
        atol=atol,
    )
