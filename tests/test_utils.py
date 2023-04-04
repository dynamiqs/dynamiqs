import qutip as qt
from pytest import approx

import torchqdynamics as tq


def test_ket_fidelity_correctness():
    n = 8
    psi = qt.rand_ket(n, seed=42)
    psi_tensor = tq.from_qutip(psi)
    phi = qt.rand_ket(n, seed=43)
    phi_tensor = tq.from_qutip(phi)

    qt_fid = qt.fidelity(psi, phi) ** 2
    tq_fid = tq.ket_fidelity(psi_tensor, phi_tensor).item()

    assert qt_fid == approx(tq_fid)


def test_ket_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    phi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    psi = tq.types.to_tensor(psi)
    phi = tq.types.to_tensor(phi)
    assert tq.ket_fidelity(psi, phi).shape == (b1, b2)


def test_dm_fidelity_correctness():
    n = 8
    rho = qt.rand_dm(n, n, seed=42)
    rho_tensor = tq.from_qutip(rho)
    sigma = qt.rand_dm(n, n, seed=43)
    sigma_tensor = tq.from_qutip(sigma)

    qt_fid = qt.fidelity(rho, sigma) ** 2
    tq_fid = tq.dm_fidelity(rho_tensor, sigma_tensor).item()

    assert qt_fid == approx(tq_fid)


def test_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    sigma = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    rho = tq.types.to_tensor(rho)
    sigma = tq.types.to_tensor(sigma)
    assert tq.dm_fidelity(rho, sigma).shape == (b1, b2)
