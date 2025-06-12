import jax
import jax.numpy as jnp
import pytest
import qutip as qt
from jax import Array

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_dag():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.dag).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_powm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.powm, static_argnums=(1,)).trace(x, 2)


@pytest.mark.run(order=TEST_INSTANT)
def test_expm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.expm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_cosm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.cosm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_sinm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.sinm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_signm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.signm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_trace():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.trace).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_tracemm():
    # prepare inputs
    keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 2)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.tracemm).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_ptrace():
    # prepare inputs
    keya, keyb, keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 4)

    a = dq.random.ket(keya, (5, 1))
    b = dq.random.ket(keyb, (8, 1))
    x = dq.random.dm(keyx, (5, 5))
    y = dq.random.dm(keyy, (8, 8))

    # check that no error is raised while tracing the function
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(a & b, 0)
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(x & y, 0)

    # test correctness
    ap = dq.ptrace(a & b, 0)
    assert jnp.allclose(a.todm().to_jax(), ap.to_jax(), 1e-5)

    xp = dq.ptrace(x & y, 0)
    assert jnp.allclose(x.to_jax(), xp.to_jax(), 1e-5)


@pytest.mark.run(order=TEST_INSTANT)
def test_tensor():
    # prepare inputs
    keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 2)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.tensor).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_expect():
    # prepare inputs
    keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 2)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.expect).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_norm():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    x = dq.random.dm(keyx, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.norm).trace(a)
    jax.jit(dq.norm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_unit():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    x = dq.random.dm(keyx, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.unit).trace(a)
    jax.jit(dq.unit).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_dissipator():
    # prepare inputs
    keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 2)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.dissipator).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_lindbladian():
    # prepare inputs
    keyx, keyy, keyz = jax.random.split(jax.random.PRNGKey(0), 3)
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))
    z = dq.random.dm(keyz, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.lindbladian).trace(x, [y], z)


@pytest.mark.run(order=TEST_INSTANT)
def test_isket():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    a = dq.random.ket(key, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.isket).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_isbra():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    a = dq.random.ket(key, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.isbra).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_isdm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.isdm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_isop():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.isop).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_isherm():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    x = dq.random.dm(key, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.isherm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_toket():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    a = dq.random.ket(key, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.toket).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_tobra():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    a = dq.random.ket(key, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.tobra).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_todm():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    x = dq.random.dm(keyx, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.todm).trace(a)
    jax.jit(dq.todm).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_proj():
    # prepare inputs
    key = jax.random.PRNGKey(0)
    a = dq.random.ket(key, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.proj).trace(a)


@pytest.mark.run(order=TEST_INSTANT)
def test_braket():
    # prepare inputs
    keya, keyb = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    b = dq.random.ket(keyb, (4, 1))

    # check that no error is raised while tracing the function
    jax.jit(dq.braket).trace(a, b)


@pytest.mark.run(order=TEST_INSTANT)
def test_overlap():
    # prepare inputs
    keya, keyb, keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 4)

    a = dq.random.ket(keya, (4, 1))
    b = dq.random.ket(keyb, (4, 1))
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.overlap).trace(a, b)
    jax.jit(dq.overlap).trace(a, y)
    jax.jit(dq.overlap).trace(x, b)
    jax.jit(dq.overlap).trace(x, y)


@pytest.mark.run(order=TEST_INSTANT)
def test_fidelity():
    # prepare inputs
    keya, keyb, keyx, keyy = jax.random.split(jax.random.PRNGKey(0), 4)

    a = dq.random.ket(keya, (4, 1))
    b = dq.random.ket(keyb, (4, 1))
    x = dq.random.dm(keyx, (4, 4))
    y = dq.random.dm(keyy, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.fidelity).trace(a, b)
    jax.jit(dq.fidelity).trace(a, y)
    jax.jit(dq.fidelity).trace(x, b)
    jax.jit(dq.fidelity).trace(x, y)

    # test correctness
    _test_ket_fidelity_correctness()
    _test_ket_fidelity_batching()
    _test_dm_fidelity_correctness()
    _test_dm_fidelity_batching()
    _test_ket_dm_fidelity_correctness()
    _test_ket_dm_fidelity_batching()


def _test_ket_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    phi = qt.rand_ket(n, seed=43)
    qt_fid = qt.fidelity(psi, phi) ** 2

    # Dynamiqs
    psi = _qobj_to_array(psi)
    phi = _qobj_to_array(phi)
    dq_fid = dq.fidelity(psi, phi).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid)


def _test_ket_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    phi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    psi = _qobj_to_array(psi)
    phi = _qobj_to_array(phi)
    assert dq.fidelity(psi, phi).shape == (b1, b2)


def _test_dm_fidelity_correctness():
    n = 8

    # qutip
    rho = qt.rand_dm(n, n, seed=42)
    sigma = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(rho, sigma) ** 2

    # Dynamiqs
    rho = _qobj_to_array(rho)
    sigma = _qobj_to_array(sigma)
    dq_fid = dq.fidelity(rho, sigma).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid, abs=1e-5)


def _test_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    sigma = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    rho = _qobj_to_array(rho)
    sigma = _qobj_to_array(sigma)
    assert dq.fidelity(rho, sigma).shape == (b1, b2)


def _test_ket_dm_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    rho = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(psi, rho) ** 2

    # Dynamiqs
    psi = _qobj_to_array(psi)
    rho = _qobj_to_array(rho)
    dq_fid_ket_dm = dq.fidelity(psi, rho).item()
    dq_fid_dm_ket = dq.fidelity(rho, psi).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid_ket_dm, abs=1e-6)
    assert qt_fid == pytest.approx(dq_fid_dm_ket, abs=1e-6)


def _test_ket_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n)
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    psi = _qobj_to_array(psi)
    rho = _qobj_to_array(rho)
    assert dq.fidelity(rho, psi).shape == (b1, b2)
    assert dq.fidelity(psi, rho).shape == (b1, b2)


def _qobj_to_array(x: qt.Qobj) -> Array:
    # todo: support QuTiP >= 5.0, remove once https://github.com/qutip/qutip/pull/2533
    # is merged, and use `jnp.asarray` instead
    if isinstance(x, list):
        return jnp.asarray([_qobj_to_array(y) for y in x])
    return jnp.asarray(x.full())


@pytest.mark.run(order=TEST_INSTANT)
def test_purity():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    x = dq.random.dm(keyx, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.purity).trace(a)
    jax.jit(dq.purity).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_entropy_vn():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (4, 1))
    x = dq.random.dm(keyx, (4, 4))

    # check that no error is raised while tracing the function
    jax.jit(dq.entropy_vn).trace(a)
    jax.jit(dq.entropy_vn).trace(x)


@pytest.mark.run(order=TEST_INSTANT)
def test_bloch_coordinates():
    # prepare inputs
    keya, keyx = jax.random.split(jax.random.PRNGKey(0), 2)
    a = dq.random.ket(keya, (2, 1))
    x = dq.random.dm(keyx, (2, 2))

    # check that no error is raised while tracing the function
    jax.jit(dq.bloch_coordinates).trace(a)
    jax.jit(dq.bloch_coordinates).trace(x)
