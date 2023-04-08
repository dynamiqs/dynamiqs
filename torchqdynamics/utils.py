from typing import List, Optional, Sequence, Tuple, Union

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
    """Check if a tensor is in state vector format."""
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


def sqrtm(x: Tensor) -> Tensor:
    """Compute the square root of a symmetric or Hermitian positive definite matrix.

    Args:
        x: Tensor of dimension `(..., n, n)`.

    Returns:
        Tensor of dimension `(..., n, n)` holding the square root of `x`.
    """
    # code copied from
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    L, Q = torch.linalg.eigh(x)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def ket_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Return the fidelity between two state vectors.

    The fidelity between two pure states $\varphi$ and $\psi$ is defined by their
    squared overlap
    $$
        F(\varphi, \psi) = |\braket{\varphi, \psi}|^2.
    $$

    Warning:
        This definition is different from QuTiP `fidelity()` function which
        uses the square root fidelity $F' = \sqrt{F}`.

    Note:
        This fidelity is also sometimes called the *Uhlmann state fidelity*.

    Args:
        x: Tensor of size `(..., n, 1)`.
        y: Tensor of size `(..., n, 1)`.

    Returns:
        Tensor of size `(...)` holding the real-valued fidelity.
    """
    return ket_overlap(x, y).abs().pow(2).real


def dm_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Return the fidelity between two density matrices.

    The fidelity between two density matrices $\rho$ and $\sigma$ is defined by
    $$
        F(\rho, \sigma) = \mathrm{Tr}\left[\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right]^2
    $$

    Warning:
        This definition is different from QuTiP `fidelity()` function which
        uses the square root fidelity $F' = \sqrt{F}`.

    Note:
        This fidelity is also sometimes called the *Uhlmann state fidelity*.

    Args:
        x: Tensor of size `(..., n, n)`.
        y: Tensor of size `(..., n, n)`.

    Returns:
        Tensor of size `(...)` holding the real-valued fidelity.
    """
    sqrtm_x = sqrtm(x)
    tmp = sqrtm_x @ y @ sqrtm_x

    # we don't need the whole matrix `sqrtm(tmp)`, just its trace, which can be computed
    # by summing the square roots of `tmp` eigenvalues
    eigvals_tmp = torch.linalg.eigvalsh(tmp)

    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    eigvals_tmp = eigvals_tmp.where(eigvals_tmp >= 0, zero)

    trace_sqrtm_tmp = torch.sqrt(eigvals_tmp).sum(-1)

    return trace_sqrtm_tmp.pow(2).real


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
    operators.

    Note:
        This function is the equivalent of `qutip.tensor`.
    """
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
    x: Tensor, keep: Union[int, Tuple[int, ...]], dims: Tuple[int, ...]
) -> Tensor:
    """Compute the partial trace of a state vector or density matrix.

    Args:
        x: Tensor of size `(..., n, 1)` or `(..., n, n)`
        keep: Int or tuple of ints containing the dimensions to keep for the
            partial trace.
        dims: Tuple of ints specifying the dimensions of each subsystem in the
            Hilbert space tensor product.

    Returns:
        Tensor of size `(..., m, m)` with `m <= n` containing the partially traced out
            state vector or density matrix.

    Example:
        >>> rho = tq.tensprod(tq.coherent_dm(20, 2.0),
                              tq.fock_dm(2, 0),
                              tq.fock_dm(5, 1))
        >>> rhoA = tq.ptrace(rho, 0, (20, 2, 5))
        >>> rhoA.shape
        torch.Size([20, 20])
        >>> rhoBC = tq.ptrace(rho, (1, 2), (20, 2, 5))
        >>> rhoBC.shape
        torch.Size([10, 10])
    """
    # convert keep and dims to tensors
    if isinstance(keep, int):
        keep = torch.as_tensor([keep])
    elif isinstance(keep, tuple):
        keep = torch.as_tensor(keep)  # e.g. [1, 2]
    dims = torch.as_tensor(dims)  # e.g. [20, 2, 5]
    ndims = len(dims)  # e.g. 3

    # check that input dimensions match
    if not torch.prod(dims) == x.size(-2):
        raise ValueError(
            f'Input `dims` {dims.tolist()} does not match the input '
            f'tensor size of {x.size(-2)}.'
        )
    if torch.any(keep < 0) or torch.any(keep > len(dims) - 1):
        raise ValueError(
            f'Input `keep` {keep.tolist()} does not match the Hilbert '
            f'space structure {dims.tolist()}.'
        )

    # sort keep
    keep = keep.sort()[0]

    # create einsum alphabet
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # compute einsum equations
    eq1 = alphabet[:ndims]  # e.g. 'abc'
    unused = iter(alphabet[ndims:])
    eq2 = [next(unused) if i in keep else eq1[i] for i in range(ndims)]  # e.g. 'ade'

    # trace out x over unkept dimensions
    if is_ket(x):
        x = x.reshape(-1, *dims)  # e.g. (..., 20, 2, 5)
        eq = ''.join(['...'] + eq1 + [',...'] + eq2)  # e.g. '...abc,...ade'
        x = torch.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    else:
        x = x.reshape(-1, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = ''.join(['...'] + eq1 + eq2)  # e.g. '...abcade'
        x = torch.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)

    # reshape to final dimension
    nkeep = torch.prod(dims[keep])  # e.g. 10
    return x.reshape(-1, nkeep, nkeep)  # e.g. (..., 10, 10)


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
