from typing import List, Optional

import torch
from qutip import Qobj
from torch import Tensor


def is_ket(x: Tensor) -> Tensor:
    return x.size(-1) == 1


def ket_to_bra(x: Tensor) -> Tensor:
    """Linear map (bra) representation of a state vector (ket).

    Args:
        x: Tensor of dimension `(..., d, 1)`.

    Returns:
        Tensor of dimension `(..., 1, d)`.
    """
    return x.adjoint()


def ket_to_dm(x: Tensor) -> Tensor:
    """Density matrix formed by the outer product of a state vector (ket).

    Args:
        x: Tensor of dimension `(..., d, 1)`.

    Returns:
        Tensor of dimension `(..., d, d)`.
    """
    return x @ ket_to_bra(x)


def ket_overlap(x: Tensor, y: Tensor) -> torch.complex:
    """Return the overlap (inner product) between two state vectors (kets).

    Args:
        x: Tensor of dimension `(..., d, 1)`.
        y: Tensor of dimension `(..., d, 1)`.

    Returns:
        Complex-valued overlap (`torch.complex64` or `torch.complex128`).
    """
    return (ket_to_bra(x) @ y).squeeze(-1).sum(-1)


def ket_fidelity(x: Tensor, y: Tensor) -> float:
    """Return the fidelity between two state vectors (kets).

    The fidelity between two pure states $\varphi$, $\psi$ is defined by their
    squared overlap
    $$
        F(\varphi, \psi) = |\braket{\varphi, \psi}|^2.
    $$

    Warning:
        This definition is different from QuTiP `fidelity()` function which
        uses the square root fidelity $F' = \sqrt{F}`.

    Args:
        x: Tensor of dimension `(..., d, 1)`.
        y: Tensor of dimension `(..., d, 1)`.

    Returns:
        Real-valued fidelity (`torch.float32` or `torch.float64`).
    """
    return ket_overlap(x, y).abs().pow(2)


def dissipator(L: Tensor, rho: Tensor) -> Tensor:
    """Apply the dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L](\cdot)$ is defined by
    $$
        \mathcal{D}[L](\rho) = L\rho L^dag - \frac{1}{2}L^\dag L \rho - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L: Jump operator of dimension `(..., d, d)`.
        rho: Density matrix of dimension `(..., d, d)`.

    Returns:
        Tensor of dimension `(..., d, d)` resulting from the application of the
        dissipation superoperator.
    """
    return (
        L @ rho @ L.adjoint() - 0.5 * L.adjoint() @ L @ rho -
        0.5 * rho @ L.adjoint() @ L
    )


def lindbladian(H: Tensor, Ls: Tensor, rho: Tensor) -> Tensor:
    """Apply the Lindbladian superoperator to a density matrix.

    The system Lindbladian $\mathcal{L}(\cdot)$ is the superoperator
    generating the evolution of the system. It is defined by
    $$
        \frac{\mathrm{d}\rho}{\mathrm{d}t} = \mathcal{L}(\rho) = -i[H,\rho] + \sum_{i=1}^n \mathcal{D}[L_i](\rho).
    $$

    Args:
        H: Hamiltonian of dimension `(..., d, d)`.
        Ls: Jump operators tensor of dimension `(..., n, d, d)`.
        rho: Density matrix of dimension `(..., d, d)`.

    Returns:
        Tensor of dimension `(..., d, d)` resulting from the application of the
        Lindbladian.
    """
    return -1j * (H @ rho - rho @ H) + dissipator(Ls, rho).sum(0)


def trace(rho: Tensor) -> Tensor:
    """Compute the batched trace of a tensor over its last two dimensions."""
    return torch.einsum('...ii', rho)


def expect(operator: Tensor, state: Tensor) -> Tensor:
    """Compute the expectation value of an operator on a quantum state or density
    matrix. The method is batchable over the state, but not over the operator.

    Args:
        operator: tensor of shape (n, n)
        state: tensor of shape (..., n, n) or (..., n)
    Returns:
        expectation value of shape (...)
    """
    # TODO: Once QTensor is implemented, check if state is a density matrix or ket.
    # For now, we assume it is a density matrix.
    return torch.einsum('ij,...ji', operator, state)


def from_qutip(x: Qobj) -> Tensor:
    """Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x: QuTiP quantum object.

    Returns:
        PyTorch tensor.
    """
    return torch.from_numpy(x.full())


def to_qutip(x: Tensor, dims: Optional[List] = None) -> Qobj:
    """Convert a PyTorch tensor to a QuTiP quantum object.

    Args:
        x: PyTorch tensor.
        dims: QuTiP object dimensions.

    Returns:
        QuTiP quantum object.
    """
    return Qobj(x.numpy(force=True), dims=dims)
