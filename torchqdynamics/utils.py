from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from qutip import Qobj
from torch import Tensor

__all__ = [
    'is_ket',
    'ket_to_bra',
    'ket_to_dm',
    'ket_overlap',
    'ket_fidelity',
    'dissipator',
    'lindbladian',
    'trace',
    'ptrace',
    'expect',
    'from_qutip',
    'to_qutip',
]


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
    r"""Return the fidelity between two state vectors (kets).

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
    r"""Apply the dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L](\cdot)$ is defined by
    $$
        \mathcal{D}[L](\rho) = L\rho L^dag - \frac{1}{2}L^\dag L \rho
                               - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L: Jump operator of dimension `(..., d, d)`.
        rho: Density matrix of dimension `(..., d, d)`.

    Returns:
        Tensor of dimension `(..., d, d)` resulting from the application of the
        dissipation superoperator.
    """
    return (
        L @ rho @ L.adjoint()
        - 0.5 * L.adjoint() @ L @ rho
        - 0.5 * rho @ L.adjoint() @ L
    )


def lindbladian(H: Tensor, Ls: Tensor, rho: Tensor) -> Tensor:
    r"""Apply the Lindbladian superoperator to a density matrix.

    The system Lindbladian $\mathcal{L}(\cdot)$ is the superoperator
    generating the evolution of the system. It is defined by
    $$
        \frac{\mathrm{d}\rho}{\mathrm{d}t} = \mathcal{L}(\rho) = -i[H,\rho]
                                             + \sum_{i=1}^n \mathcal{D}[L_i](\rho).
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


def kron(*x: Tensor):
    """Compute the tensor product of a sequence of state vectors, density matrices or
    operators."""
    x = _extract_tuple_from_varargs(x)
    y = x[0]
    for _x in x[1:]:
        y = torch.kron(y, _x)
    return y


def _extract_tuple_from_varargs(x: Union[Tuple, Tuple[Tuple]]) -> Tuple:
    """Returns a tuple from varargs.

    This copies the behavior of PyTorch which accepts both varargs as `foo(1,2,3)` or
    `foo((1,2,3,))`.
    """
    # Check tuple is not empty
    if len(x) == 0:
        raise TypeError('No arguments were supplied.')

    # Handles tuple unwrapping
    if len(x) == 1 and isinstance(x[0], Sequence):
        x = x[0]

    return x


def trace(rho: Tensor) -> Tensor:
    """Compute the batched trace of a tensor over its last two dimensions."""
    return torch.einsum('...ii', rho)


def ptrace(
    x: Tensor, dims_kept: Union[int, Tuple[int, ...]], hilbert_shape: Tuple[int, ...]
) -> Tensor:
    """Compute the partial trace of a state vector or density matrix, keeping only
    dimensions `dims_kept`. The Hilbert space structure should be specified with
    `hilbert_shape`.

    Args:
        x: Tensor of size `(..., n, 1)` or `(..., n, n)`
        dims_kept: Int or tuple of ints containing the dimensions to keep for the
            partial trace.
        hilbert_shape: Tuple of ints specifying the dimensions of each mode in the
            Hilbert space tensor product.

    Returns:
        Tensor of size `(..., m, m)` with `m <= n` containing the partially traced out
            state vector or density matrix.

    Example:
        >>> rho = tq.kron(tq.coherent_dm(20, 2.0), tq.fock_dm(2, 0), tq.fock_dm(3, 1))
        >>> rhoA = tq.ptrace(rho, 0, (20, 2, 3))
        >>> rhoA.shape
        torch.Size([20, 20])
        >>> rhoBC = tq.ptrace(rho, (1, 2), (20, 2, 3))
        >>> rhoBC.shape
        torch.Size([6, 6])
    """
    # convert dims_kept and hilbert_shape to tensors
    hilbert_shape = torch.as_tensor(hilbert_shape)
    if isinstance(dims_kept, int):
        dims_kept = torch.as_tensor([dims_kept])
    elif isinstance(dims_kept, tuple):
        dims_kept = torch.as_tensor(dims_kept)

    # check that input dimensions match
    if not torch.prod(hilbert_shape) == x.size(-2):
        raise ValueError(
            f'Input `hilbert_shape` {hilbert_shape} does not match the input tensor'
            f' size of {x.size(-2)}.'
        )
    if torch.any(dims_kept < 0) or torch.any(dims_kept > len(hilbert_shape) - 1):
        raise ValueError(
            f'The specified dimension {dims_kept} does not match the Hilbert space'
            f' structure {hilbert_shape}.'
        )

    # sort dims_kept
    dims_kept = dims_kept.sort()[0]

    # find dimensions to trace out
    ndims = len(hilbert_shape)
    dims = torch.arange(0, ndims)
    dims_trace = torch.as_tensor(np.setdiff1d(dims, dims_kept))

    # find sizes to keep and trace out
    size_kept = torch.prod(hilbert_shape[dims_kept])
    size_trace = torch.prod(hilbert_shape[dims_trace])

    # find batch shape
    ndims_b = x.ndim - 2
    dims_b = torch.arange(0, ndims_b)
    b_shape = x.shape[:-2]

    if is_ket(x):
        # find permutation that puts traced out dimensions last
        perm = tuple(dims_b) + tuple(dims_kept + ndims_b) + tuple(dims_trace + ndims_b)

        # reshape to the Hilbert space shape, permute and reshape again
        x = x.reshape(*b_shape, *hilbert_shape)
        x = x.permute(perm)
        x = x.reshape(*b_shape, size_kept, 1, size_trace)
        y = x.transpose(-2, -3).conj()

        # trace out last dimension
        return torch.linalg.vecdot(x, y)
    else:
        # find permutation that puts traced out dimensions last
        perm = (
            tuple(dims_b)
            + tuple(dims_kept + ndims_b)
            + tuple(dims_kept + ndims_b + ndims)
            + tuple(dims_trace + ndims_b)
            + tuple(dims_trace + ndims_b + ndims)
        )

        # reshape to the Hilbert space shape, permute and reshape again
        x = x.reshape(*b_shape, *hilbert_shape, *hilbert_shape)
        x = x.permute(perm)
        x = x.reshape(*b_shape, size_kept, size_kept, size_trace, size_trace)

        # trace out the last dimensions
        return trace(x)


def expect(O: Tensor, x: Tensor) -> Tensor:
    r"""Compute the expectation values of an operator on a state vector or a density
    matrix.

    The expectation value $\braket{O}$ of a single operator $O$ is computed
    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a state vector $\psi$,
    - as $\braket{O}=\tr(O\rho)$ if `x` is a density matrix $\rho$.

    Note:
        The returned tensor is complex-valued.

    TODO Adapt to bras.

    Args:
        O: Tensor of size `(n, n)`
        x: Tensor of size `(..., n, 1)` or `(..., n, n)`

    Returns:
        Tensor of size `(...)` holding the operator expectation values.
        expectation value of size (...)
    """
    if is_ket(x):
        return torch.einsum('...ij,jk,...kl->...', x.adjoint(), O, x)  # <x|O|x>
    return torch.einsum('ij,...ji->...', O, x)  # tr(Ox)


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
