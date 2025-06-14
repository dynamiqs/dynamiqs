import jax
import jax.numpy as jnp
import pytest
import qutip as qt

import dynamiqs as dq

from ..order import TEST_INSTANT, TEST_SHORT

# prepare inputs
key = jax.random.PRNGKey(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
a = pytest.fixture(lambda: dq.random.ket(k1, (4, 1)))
b = pytest.fixture(lambda: dq.random.ket(k2, (4, 1)))
x = pytest.fixture(lambda: dq.random.dm(k3, (4, 4)))
y = pytest.fixture(lambda: dq.random.dm(k4, (4, 4)))
z = pytest.fixture(lambda: dq.random.dm(k5, (4, 4)))


@pytest.mark.run(order=TEST_INSTANT)
def test_dag(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.dag).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_powm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.powm, static_argnums=(1,)).trace(x, 2)


@pytest.mark.run(order=TEST_INSTANT)
def test_expm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.expm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_cosm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.cosm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_sinm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.sinm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_signm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.signm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_trace(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.trace).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_tracemm(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.tracemm).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_ptrace():
    # prepare inputs
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    a = dq.random.ket(k1, (5, 1))
    b = dq.random.ket(k2, (8, 1))
    x = dq.random.dm(k3, (5, 5))
    y = dq.random.dm(k4, (8, 8))

    # check that no error is raised while tracing the function
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(a & b, 0)
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(x & y, 0)

    # test correctness
    ap = dq.ptrace(a & b, 0)
    assert jnp.allclose(a.todm().to_jax(), ap.to_jax())

    xp = dq.ptrace(x & y, 0)
    assert jnp.allclose(x.to_jax(), xp.to_jax())


@pytest.mark.run(order=TEST_INSTANT)
def test_tensor(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.tensor).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_expect(a, x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.expect).trace(x, a)
    jax.jit(dq.expect).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_norm(a, x):
    # check that no error is raised while tracing the function
    jax.jit(dq.norm).trace(a)
    jax.jit(dq.norm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_unit(a, x):
    # check that no error is raised while tracing the function
    jax.jit(dq.unit).trace(a)
    jax.jit(dq.unit).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_dissipator(x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.dissipator).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_lindbladian(x, y, z):
    # check that no error is raised while tracing the function
    jax.jit(dq.lindbladian).trace(x, [], z)
    jax.jit(dq.lindbladian).trace(x, [y], z)
    jax.jit(dq.lindbladian).trace(x, [y, y], z)


@pytest.mark.run(order=TEST_INSTANT)
def test_isket(a):
    # check that no error is raised while tracing the function
    jax.jit(dq.isket).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_isbra(a):
    # check that no error is raised while tracing the function
    jax.jit(dq.isbra).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_isdm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.isdm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_isop(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.isop).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_isherm(x):
    # check that no error is raised while tracing the function
    jax.jit(dq.isherm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_toket(a):
    # check that no error is raised while tracing the function
    jax.jit(dq.toket).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_tobra(a):
    # check that no error is raised while tracing the function
    jax.jit(dq.tobra).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_todm(a, x):
    # check that no error is raised while tracing the function
    jax.jit(dq.todm).trace(a)
    jax.jit(dq.todm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_proj(a):
    # check that no error is raised while tracing the function
    jax.jit(dq.proj).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_braket(a, b):
    # check that no error is raised while tracing the function
    jax.jit(dq.braket).trace(a, b)


@pytest.mark.run(order=TEST_INSTANT)
def test_overlap(a, b, x, y):
    # check that no error is raised while tracing the function
    jax.jit(dq.overlap).trace(a, b)
    jax.jit(dq.overlap).trace(a, y)
    jax.jit(dq.overlap).trace(x, b)
    jax.jit(dq.overlap).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_fidelity_tracing(a, b, x, y):
    # === check that no error is raised while tracing the function
    jax.jit(dq.fidelity).trace(a, b)
    jax.jit(dq.fidelity).trace(a, y)
    jax.jit(dq.fidelity).trace(x, b)
    jax.jit(dq.fidelity).trace(x, y)


@pytest.mark.run(order=TEST_SHORT)
def test_fidelity_correctness(a, b, x, y):
    # ket vs ket, dm vs dm, ket vs dm
    for X, Y in [(a, b), (x, y), (a, x)]:
        qt_fid = qt.fidelity(X.to_qutip(), Y.to_qutip()) ** 2
        dq_fid = dq.fidelity(X, Y).item()
        assert qt_fid == pytest.approx(dq_fid, rel=1e-5, abs=1e-5)


@pytest.mark.run(order=TEST_SHORT)
def test_fidelity_batching(a, b, x, y):
    b1, b2 = 3, 5
    batch = lambda X: dq.asqarray(jnp.tile(X.to_jax(), (3, 5, 1, 1)))
    assert dq.fidelity(batch(a), batch(b)).shape == (b1, b2)
    assert dq.fidelity(batch(x), batch(y)).shape == (b1, b2)
    assert dq.fidelity(batch(a), batch(y)).shape == (b1, b2)


@pytest.mark.run(order=TEST_INSTANT)
def test_purity(a, x):
    # check that no error is raised while tracing the function
    jax.jit(dq.purity).trace(a)
    jax.jit(dq.purity).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_entropy_vn(a, x):
    # check that no error is raised while tracing the function
    jax.jit(dq.entropy_vn).trace(a)
    jax.jit(dq.entropy_vn).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_bloch_coordinates():
    # prepare inputs
    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    a = dq.random.ket(k1, (2, 1))
    x = dq.random.dm(k2, (2, 2))

    # check that no error is raised while tracing the function
    jax.jit(dq.bloch_coordinates).trace(a)
    jax.jit(dq.bloch_coordinates).trace(x)
