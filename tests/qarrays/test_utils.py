import jax.numpy as jnp
import jax.random as jr

import dynamiqs as dq


def test_stack_simple(rtol=1e-05, atol=1e-08):
    n = 10
    key = jr.key(42)

    jax_m = jr.uniform(key, (n, n))
    dense_m = dq.asqarray(jax_m)
    sparse_m = dq.to_sparse_dia(jax_m)

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


def test_stack_double(rtol=1e-05, atol=1e-08):
    n = 10
    key = jr.key(42)
    k1, k2 = jr.split(key)

    jax_m1 = jr.uniform(k1, (n, n))
    dense_m1, sparse_m1 = dq.asqarray(jax_m1), dq.to_sparse_dia(jax_m1)
    dense_m1 = dq.stack([dense_m1, 2 * dense_m1, 3 * dense_m1])
    sparse_m1 = dq.stack([sparse_m1, 2 * sparse_m1, 3 * sparse_m1])
    jax_m1 = jnp.stack([jax_m1, 2 * jax_m1, 3 * jax_m1])

    jax_m2 = jr.uniform(k2, (n, n))
    dense_m2, sparse_m2 = dq.asqarray(jax_m2), dq.to_sparse_dia(jax_m2)
    dense_m2 = dq.stack([dense_m2, 2 * dense_m2, 3 * dense_m2])
    sparse_m2 = dq.stack([sparse_m2, 2 * sparse_m2, 3 * sparse_m2])
    jax_m2 = jnp.stack([jax_m2, 2 * jax_m2, 3 * jax_m2])

    assert jnp.allclose(
        dq.stack([dense_m1, dense_m2]).to_jax(),
        jnp.stack([jax_m1, jax_m2]),
        rtol=rtol,
        atol=atol,
    )

    assert jnp.allclose(
        dq.stack([sparse_m1, sparse_m2]).to_jax(),
        jnp.stack([jax_m1, jax_m2]),
        rtol=rtol,
        atol=atol,
    )