import jax
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_tracing():
    # prepare inputs
    keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 2)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (3, 4, 4))

    # check that no error is raised while tracing the functions
    jax.jit(dq.vectorize)(x)
    jax.jit(dq.vectorize)(y)
    jax.jit(dq.unvectorize)(dq.vectorize(x))
    jax.jit(dq.unvectorize)(dq.vectorize(y))
    jax.jit(dq.spre)(x)
    jax.jit(dq.spre)(y)
    jax.jit(dq.spost)(x)
    jax.jit(dq.spost)(y)
    jax.jit(dq.sprepost)(x, x)
    jax.jit(dq.sprepost)(x, y)
    jax.jit(dq.sprepost)(y, x)
    jax.jit(dq.sprepost)(y, y)
    jax.jit(dq.sdissipator)(x)
    jax.jit(dq.sdissipator)(y)
    jax.jit(dq.slindbladian)(x, [x])
    jax.jit(dq.slindbladian)(y, [x])
    jax.jit(dq.slindbladian)(x, [y])
    jax.jit(dq.slindbladian)(y, [y])
