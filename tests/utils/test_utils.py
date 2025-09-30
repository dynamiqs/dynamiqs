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
def test_entropy_relative_tracing(a, b, x, y):
    # === check that no error is raised while tracing the function
    jax.jit(dq.entropy_relative).trace(a, b)  # ket vs ket
    jax.jit(dq.entropy_relative).trace(x, y)  # dm vs dm
    jax.jit(dq.entropy_relative).trace(a, x)  # ket vs dm
    jax.jit(dq.entropy_relative).trace(x, b)  # dm vs ket


@pytest.mark.run(order=TEST_SHORT)
def test_entropy_relative_correctness_against_qutip():
    n = 8

    # --- ket vs ket
    psi_qt = qt.rand_ket(n, seed=42)
    phi_qt = qt.rand_ket(n, seed=43)
    qt_val = qt.entropy_relative(psi_qt, phi_qt)

    psi = qobj_to_array(psi_qt)
    phi = qobj_to_array(phi_qt)
    dq_val = dq.entropy_relative(psi, phi).item()
    assert qt_val == pytest.approx(dq_val, rel=1e-6, abs=1e-6)

    # --- dm vs dm
    rho_qt = qt.rand_dm(n, n, seed=44)
    sigma_qt = qt.rand_dm(n, n, seed=45)
    qt_val = qt.entropy_relative(rho_qt, sigma_qt)

    rho = qobj_to_array(rho_qt)
    sigma = qobj_to_array(sigma_qt)
    dq_val = dq.entropy_relative(rho, sigma).item()
    assert qt_val == pytest.approx(dq_val, rel=1e-5, abs=1e-5)

    # --- ket vs dm and dm vs ket
    psi_qt = qt.rand_ket(n, seed=46)
    rho_qt = qt.rand_dm(n, n, seed=47)
    qt_ket_dm = qt.entropy_relative(psi_qt, rho_qt)
    qt_dm_ket = qt.entropy_relative(rho_qt, psi_qt)

    psi = qobj_to_array(psi_qt)
    rho = qobj_to_array(rho_qt)
    dq_ket_dm = dq.entropy_relative(psi, rho).item()
    dq_dm_ket = dq.entropy_relative(rho, psi).item()

    assert qt_ket_dm == pytest.approx(dq_ket_dm, rel=1e-6, abs=1e-6)
    assert qt_dm_ket == pytest.approx(dq_dm_ket, rel=1e-6, abs=1e-6)


@pytest.mark.run(order=TEST_SHORT)
def test_entropy_relative_batching(a, b, x, y):
    b1, b2 = 3, 5

    # Same batching trick you used for fidelity
    batch = lambda X: dq.asqarray(jnp.tile(X.to_jax(), (b1, b2, 1, 1)))

    # ket vs ket, dm vs dm, ket vs dm, dm vs ket
    assert dq.entropy_relative(batch(a), batch(b)).shape == (b1, b2)
    assert dq.entropy_relative(batch(x), batch(y)).shape == (b1, b2)
    assert dq.entropy_relative(batch(a), batch(x)).shape == (b1, b2)
    assert dq.entropy_relative(batch(x), batch(b)).shape == (b1, b2)


@pytest.mark.run(order=TEST_SHORT)
def test_entropy_relative_doc_examples():
    # 1) Identity case: S(rho || rho) = 0
    rho = dq.fock_dm(2, 0)
    val = dq.entropy_relative(rho, rho).item()
    assert val == pytest.approx(0.0, abs=1e-12)

    # 2) Pure vs maximally mixed (I/2): S(|0><0| || I/2) = ln 2
    psi = dq.fock_dm(2, 0)
    maximally_mixed = (dq.fock_dm(2, 0) + dq.fock_dm(2, 1)).unit()  # I/2
    val = dq.entropy_relative(psi, maximally_mixed).item()
    assert val == pytest.approx(jnp.log(2.0), rel=1e-12, abs=1e-12)

    # and asymmetry: S(I/2 || |0><0|) = +inf (support mismatch)
    val = dq.entropy_relative(maximally_mixed, psi).item()
    assert jnp.isposinf(val)

    # 3) Pure |0⟩ vs diagonal σ with eigenvalue 1/√2 on |0⟩:
    #    S = -log(1/√2) = log √2 = 0.5 * log 2
    w0 = 2 ** (-0.5)  # 1/sqrt(2)
    sigma = w0 * dq.fock_dm(2, 0) + (1.0 - w0) * dq.fock_dm(2, 1)
    val = dq.entropy_relative(dq.fock_dm(2, 0), sigma).item()
    assert val == pytest.approx(0.5 * jnp.log(2.0), rel=1e-12, abs=1e-12)

    # 4) Orthogonal pure states: S(|1><1| || |0><0|) = +inf
    rho = dq.fock_dm(2, 1)
    sigma = dq.fock_dm(2, 0)
    val = dq.entropy_relative(rho, sigma).item()
    assert jnp.isposinf(val)


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


@pytest.mark.skip('broken test')
@pytest.mark.run(order=TEST_INSTANT)
def test_jit_ptrace():
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # kets

    # TODO: this one doesn't pass

    a = dq.random.ket(20, key=key1)
    b = dq.random.ket(30, key=key2)

    ab = a & b
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a, ap, 1e-3)

    # density matrix

    a = dq.random.dm(20, key=key3)
    b = dq.random.dm(30, key=key4)

    ab = a & b
    ap = dq.ptrace(ab, 0, (20, 30))

    assert jnp.allclose(a, ap, 1e-3)
