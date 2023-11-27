import qutip as qt
import torch
from pytest import approx

import dynamiqs as dq


def test_ket_fidelity_correctness():
    n = 8
    psi = qt.rand_ket(n, seed=42)
    psi_tensor = dq.from_qutip(psi)
    phi = qt.rand_ket(n, seed=43)
    phi_tensor = dq.from_qutip(phi)

    qt_fid = qt.fidelity(psi, phi) ** 2
    dq_fid = dq.fidelity(psi_tensor, phi_tensor).item()

    assert qt_fid == approx(dq_fid)


def test_ket_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    phi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, 1)
    psi = dq.to_tensor(psi)
    phi = dq.to_tensor(phi)
    assert dq.fidelity(psi, phi).shape == (b1, b2)


def test_dm_fidelity_correctness():
    n = 8
    rho = qt.rand_dm(n, n, seed=42)
    rho_tensor = dq.from_qutip(rho)
    sigma = qt.rand_dm(n, n, seed=43)
    sigma_tensor = dq.from_qutip(sigma)

    qt_fid = qt.fidelity(rho, sigma) ** 2
    dq_fid = dq.fidelity(rho_tensor, sigma_tensor).item()

    assert qt_fid == approx(dq_fid, abs=1e-6)


def test_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    sigma = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    rho = dq.to_tensor(rho)
    sigma = dq.to_tensor(sigma)
    assert dq.fidelity(rho, sigma).shape == (b1, b2)


def test_ket_dm_fidelity_correctness():
    n = 8
    psi = qt.rand_ket(n, seed=42)
    psi_tensor = dq.from_qutip(psi)
    rho = qt.rand_dm(n, n, seed=43)
    rho_tensor = dq.from_qutip(rho)

    qt_fid = qt.fidelity(psi, rho) ** 2
    dq_fid_ket_dm = dq.fidelity(psi_tensor, rho_tensor).item()
    dq_fid_dm_ket = dq.fidelity(rho_tensor, psi_tensor).item()

    assert qt_fid == approx(dq_fid_ket_dm, abs=1e-6)
    assert qt_fid == approx(dq_fid_dm_ket, abs=1e-6)


def test_ket_dm_fidelity_batching():
    b1, b2, n = 3, 5, 8
    psi = [[qt.rand_ket(n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n)
    rho = [[qt.rand_dm(n, n) for _ in range(b2)] for _ in range(b1)]  # (b1, b2, n, n)
    psi = dq.to_tensor(psi)
    rho = dq.to_tensor(rho)
    assert dq.fidelity(rho, psi).shape == (b1, b2)
    assert dq.fidelity(psi, rho).shape == (b1, b2)


def test_hadamard():
    c64 = torch.complex64

    # one qubit
    H1 = 2 ** (-1 / 2) * torch.tensor([[1, 1], [1, -1]], dtype=c64)
    assert torch.allclose(dq.hadamard(1), H1)

    # two qubits
    H2 = 0.5 * torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=c64,
    )
    assert torch.allclose(dq.hadamard(2), H2)

    # three qubits
    H3 = 2 ** (-3 / 2) * torch.tensor(
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
    assert torch.allclose(dq.hadamard(3), H3)
