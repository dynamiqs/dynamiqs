import jax
import jax.numpy as jnp
import pytest
import qutip as qt
from jax import Array

import dynamiqs as dq
from dynamiqs._utils import cdtype

from ..order import TEST_INSTANT


def qobj_to_array(x: qt.Qobj) -> Array:
    # todo: support QuTiP >= 5.0, remove once https://github.com/qutip/qutip/pull/2533
    # is merged, and use `jnp.asarray` instead
    if isinstance(x, list):
        return jnp.asarray([qobj_to_array(y) for y in x])
    return jnp.asarray(x.full())


@pytest.mark.run(order=TEST_INSTANT)
def test_ket_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    phi = qt.rand_ket(n, seed=43)
    qt_fid = qt.fidelity(psi, phi) ** 2

    # Dynamiqs
    psi = qobj_to_array(psi)
    phi = qobj_to_array(phi)
    dq_fid = dq.fidelity(psi, phi).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid)


@pytest.mark.run(order=TEST_INSTANT)
def test_ket_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    phi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    psi = qobj_to_array(psi)
    phi = qobj_to_array(phi)
    assert dq.fidelity(psi, phi).shape == (b1, b2)


@pytest.mark.run(order=TEST_INSTANT)
def test_dm_fidelity_correctness():
    n = 8

    # qutip
    rho = qt.rand_dm(n, n, seed=42)
    sigma = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(rho, sigma) ** 2

    # Dynamiqs
    rho = qobj_to_array(rho)
    sigma = qobj_to_array(sigma)
    dq_fid = dq.fidelity(rho, sigma).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid, abs=1e-5)


@pytest.mark.run(order=TEST_INSTANT)
def test_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    sigma = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    rho = qobj_to_array(rho)
    sigma = qobj_to_array(sigma)
    assert dq.fidelity(rho, sigma).shape == (b1, b2)


@pytest.mark.run(order=TEST_INSTANT)
def test_ket_dm_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    rho = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(psi, rho) ** 2

    # Dynamiqs
    psi = qobj_to_array(psi)
    rho = qobj_to_array(rho)
    dq_fid_ket_dm = dq.fidelity(psi, rho).item()
    dq_fid_dm_ket = dq.fidelity(rho, psi).item()

    # compare
    assert qt_fid == pytest.approx(dq_fid_ket_dm, abs=1e-6)
    assert qt_fid == pytest.approx(dq_fid_dm_ket, abs=1e-6)


@pytest.mark.run(order=TEST_INSTANT)
def test_ket_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n)
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    psi = qobj_to_array(psi)
    rho = qobj_to_array(rho)
    assert dq.fidelity(rho, psi).shape == (b1, b2)
    assert dq.fidelity(psi, rho).shape == (b1, b2)


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


@pytest.mark.run(order=TEST_INSTANT)
def test_jit_ptrace():
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # kets
    a = dq.random.ket(key1, (20, 1))
    b = dq.random.ket(key2, (30, 1))

    ab = a & b
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a.todm().to_jax(), ap.to_jax(), 1e-3)

    # density matrices
    a = dq.random.dm(key3, (20, 20))
    b = dq.random.dm(key4, (30, 30))

    ab = a & b
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a.to_jax(), ap.to_jax(), 1e-3)


@pytest.mark.run(order=TEST_INSTANT)
def test_jit_general_utils():
    # prepare inputs
    keya, keyb, keyx, keyy, keyz = jax.random.split(jax.random.PRNGKey(0), 5)

    a = dq.random.ket(keya, (2, 1))
    b = dq.random.ket(keyb, (2, 1))
    x = dq.random.dm(keyx, (2, 2))
    y = dq.random.dm(keyy, (2, 2))
    z = dq.random.dm(keyz, (2, 2))

    # check that no error is raised while tracing the functions
    jax.jit(dq.dag).trace(x)
    jax.jit(dq.powm, static_argnums=(1,)).trace(x, 2)
    jax.jit(dq.expm).trace(x)
    jax.jit(dq.cosm).trace(x)
    jax.jit(dq.sinm).trace(x)
    jax.jit(dq.signm).trace(x)
    jax.jit(dq.trace).trace(x)
    jax.jit(dq.tracemm).trace(x, y)
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(a & b, 0)
    jax.jit(dq.ptrace, static_argnums=(1,)).trace(x & y, 0)
    jax.jit(dq.tensor).trace(x, y)
    jax.jit(dq.expect).trace(x, y)
    jax.jit(dq.norm).trace(a)
    jax.jit(dq.norm).trace(x)
    jax.jit(dq.unit).trace(a)
    jax.jit(dq.unit).trace(x)
    jax.jit(dq.dissipator).trace(x, y)
    jax.jit(dq.lindbladian).trace(x, [y], z)
    jax.jit(dq.isket).trace(a)
    jax.jit(dq.isbra).trace(a)
    jax.jit(dq.isdm).trace(x)
    jax.jit(dq.isop).trace(x)
    jax.jit(dq.isherm).trace(x)
    jax.jit(dq.toket).trace(a)
    jax.jit(dq.tobra).trace(a)
    jax.jit(dq.todm).trace(a)
    jax.jit(dq.todm).trace(x)
    jax.jit(dq.proj).trace(a)
    jax.jit(dq.braket).trace(a, b)
    jax.jit(dq.overlap).trace(a, b)
    jax.jit(dq.overlap).trace(a, y)
    jax.jit(dq.overlap).trace(x, b)
    jax.jit(dq.overlap).trace(x, y)
    jax.jit(dq.fidelity).trace(a, b)
    jax.jit(dq.fidelity).trace(a, y)
    jax.jit(dq.fidelity).trace(x, b)
    jax.jit(dq.fidelity).trace(x, y)
    jax.jit(dq.purity).trace(a)
    jax.jit(dq.purity).trace(x)
    jax.jit(dq.entropy_vn).trace(a)
    jax.jit(dq.entropy_vn).trace(x)
    jax.jit(dq.bloch_coordinates).trace(a)
    jax.jit(dq.bloch_coordinates).trace(x)
