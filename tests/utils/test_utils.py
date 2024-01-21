import jax
import jax.numpy as jnp
import qutip as qt
from pytest import approx

import dynamiqs as dq


def test_ket_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    phi = qt.rand_ket(n, seed=43)
    qt_fid = qt.fidelity(psi, phi) ** 2

    # dynamiqs
    psi = jnp.asarray(psi)
    phi = jnp.asarray(phi)
    dq_fid = dq.fidelity(psi, phi).item()

    # compare
    assert qt_fid == approx(dq_fid)


def test_ket_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    phi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    psi = jnp.asarray(psi)
    phi = jnp.asarray(phi)
    assert dq.fidelity(psi, phi).shape == (b1, b2)


def test_dm_fidelity_correctness():
    n = 8

    # qutip
    rho = qt.rand_dm(n, n, seed=42)
    sigma = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(rho, sigma) ** 2

    # dynamiqs
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    dq_fid = dq.fidelity(rho, sigma).item()

    # compare
    assert qt_fid == approx(dq_fid, abs=1e-5)


def test_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    sigma = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    assert dq.fidelity(rho, sigma).shape == (b1, b2)


def test_ket_dm_fidelity_correctness():
    n = 8

    # qutip
    psi = qt.rand_ket(n, seed=42)
    rho = qt.rand_dm(n, n, seed=43)
    qt_fid = qt.fidelity(psi, rho) ** 2

    # dynmaiqs
    psi = jnp.asarray(psi)
    rho = jnp.asarray(rho)
    dq_fid_ket_dm = dq.fidelity(psi, rho).item()
    dq_fid_dm_ket = dq.fidelity(rho, psi).item()

    # compare
    assert qt_fid == approx(dq_fid_ket_dm, abs=1e-6)
    assert qt_fid == approx(dq_fid_dm_ket, abs=1e-6)


def test_ket_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n)
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    psi = jnp.asarray(psi)
    rho = jnp.asarray(rho)
    assert dq.fidelity(rho, psi).shape == (b1, b2)
    assert dq.fidelity(psi, rho).shape == (b1, b2)


def test_hadamard():
    c64 = jnp.complex64

    # one qubit
    H1 = 2 ** (-1 / 2) * jnp.array([[1, 1], [1, -1]], dtype=c64)
    assert jnp.allclose(dq.hadamard(1), H1)

    # two qubits
    H2 = 0.5 * jnp.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=c64,
    )
    assert jnp.allclose(dq.hadamard(2), H2)

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
        dtype=c64,
    )
    assert jnp.allclose(dq.hadamard(3), H3)


def test_jit_ptrace():
    import jax

    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # kets

    # todo: this one doesn't pass

    a = dq.rand.ket(20, key=key1)
    b = dq.rand.ket(30, key=key2)

    ab = dq.tensor(a, b)
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a, ap, 1e-3)

    # density matrix

    a = dq.rand.dm(20, key=key3)
    b = dq.rand.dm(30, key=key4)

    ab = dq.tensor(a, b)
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a, ap, 1e-3)
