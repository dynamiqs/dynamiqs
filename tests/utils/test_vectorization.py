import jax
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT

# prepare inputs
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key, 2)
x = pytest.fixture(lambda: dq.random.dm(k1, (4, 4)))
y = pytest.fixture(lambda: dq.random.dm(k2, (3, 4, 4)))


@pytest.mark.run(order=TEST_INSTANT)
def test_vectorize(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.vectorize)(x)
    jax.jit(dq.vectorize)(y)


@pytest.mark.run(order=TEST_INSTANT)
def test_unvectorize(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.unvectorize)(dq.vectorize(x))
    jax.jit(dq.unvectorize)(dq.vectorize(y))


@pytest.mark.run(order=TEST_INSTANT)
def test_spre(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.spre)(x)
    jax.jit(dq.spre)(y)


@pytest.mark.run(order=TEST_INSTANT)
def test_spost(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.spost)(x)
    jax.jit(dq.spost)(y)


@pytest.mark.run(order=TEST_INSTANT)
def test_sprepost(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.sprepost)(x, x)
    jax.jit(dq.sprepost)(x, y)
    jax.jit(dq.sprepost)(y, x)
    jax.jit(dq.sprepost)(y, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_sdissipator(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.sdissipator)(x)
    jax.jit(dq.sdissipator)(y)


@pytest.mark.run(order=TEST_INSTANT)
def test_slindbladian(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.slindbladian)(x, [])
    jax.jit(dq.slindbladian)(x, [x])
    jax.jit(dq.slindbladian)(x, [y])
    jax.jit(dq.slindbladian)(x, [x, y])

    jax.jit(dq.slindbladian)(y, [])
    jax.jit(dq.slindbladian)(y, [x])
    jax.jit(dq.slindbladian)(y, [y])
    jax.jit(dq.slindbladian)(y, [x, y])

    jax.jit(dq.slindbladian)(x, [])
    jax.jit(dq.slindbladian)(x, [x])
    jax.jit(dq.slindbladian)(x, [y])
    jax.jit(dq.slindbladian)(x, [x, y])

    jax.jit(dq.slindbladian)(y, [])
    jax.jit(dq.slindbladian)(y, [x])
    jax.jit(dq.slindbladian)(y, [y])
    jax.jit(dq.slindbladian)(y, [x, y])
