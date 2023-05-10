from __future__ import annotations

from functools import reduce

import torch
from torch import Tensor

__all__ = [
    'is_ket',
    'ket_to_bra',
    'ket_to_dm',
    'ket_overlap',
    'ket_fidelity',
    'dm_fidelity',
    'sqrtm',
    'dissipator',
    'lindbladian',
    'tensprod',
    'trace',
    'ptrace',
    'expect',
]


def is_ket(x: Tensor) -> bool:
    """Check if a tensor is in state vector format.

    Args:
        x ( Tensor (..., d, n) ): State vector, linear map or density matrix.

    Returns:
        `True` if the last dimension `n` is `1`, `False` otherwise.
    """
    return x.size(-1) == 1


def ket_to_bra(x: Tensor) -> Tensor:
    r"""Linear map $\bra\psi$ representation of a state vector $\ket\psi$.

    Args:
        x ( Tensor (..., d, 1) ): State vector.

    Returns:
        ( Tensor (..., 1, d) ): Corresponding linear map.
    """
    return x.adjoint()


def ket_to_dm(x: Tensor) -> Tensor:
    r"""Density matrix $\ket\psi\bra\psi$ formed by the outer product of a state vector
    $\ket\psi$.

    Args:
        x ( Tensor (..., d, 1) ): State vector.

    Returns:
        ( Tensor (..., d, d) ): Corresponding density matrix.
    """
    return x @ ket_to_bra(x)


def ket_overlap(x: Tensor, y: Tensor) -> Tensor:
    r"""Overlap (inner product) $\braket{\varphi,\psi}$ between two state vectors
    $\ket\varphi$ and $\ket\psi$.

    Args:
        x ( Tensor (..., d, 1) ): First state vector.
        y ( Tensor (..., d, 1) ): Second state vector.

    Returns:
        ( Tensor (...) ): Complex-valued overlap.
    """
    return (ket_to_bra(x) @ y).squeeze(-1).sum(-1)


def ket_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Fidelity between two state vectors.

    The fidelity between two pure states $\ket\varphi$ and $\ket\psi$ is defined by
    their squared overlap:

    $$
        F(\ket\varphi, \ket\psi) = \left|\braket{\varphi, \psi}\right|^2.
    $$

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Note:
        This fidelity is also sometimes called the *Uhlmann state fidelity*.

    Args:
        x ( Tensor (..., d, 1) ): First state vector.
        y ( Tensor (..., d, 1) ): Second state vector.

    Returns:
        ( Tensor (...) ): Real-valued fidelity.
    """
    return ket_overlap(x, y).abs().pow(2).real


def dm_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Fidelity between two density matrices.

    The fidelity between two density matrices $\rho$ and $\sigma$ is defined by:

    $$
        F(\rho, \sigma) = \tr{\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}}^2
    $$

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Note:
        This fidelity is also sometimes called the *Uhlmann state fidelity*.

    Args:
        x ( Tensor (..., d, d) ): First density matrix.
        y ( Tensor (..., d, d) ): Second density matrix.

    Returns:
        ( Tensor (...) ): Real-valued fidelity.
    """
    sqrtm_x = sqrtm(x)
    tmp = sqrtm_x @ y @ sqrtm_x

    # we don't need the whole matrix `sqrtm(tmp)`, just its trace, which can be computed
    # by summing the square roots of `tmp` eigenvalues
    eigvals_tmp = torch.linalg.eigvalsh(tmp)

    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    eigvals_tmp = eigvals_tmp.where(eigvals_tmp >= 0, zero)

    trace_sqrtm_tmp = torch.sqrt(eigvals_tmp).sum(-1)

    return trace_sqrtm_tmp.pow(2).real


def sqrtm(x: Tensor) -> Tensor:
    """Square root of a symmetric or Hermitian positive definite matrix.

    Args:
        x ( Tensor (..., d, d) ): Symmetric or Hermitian positive definite matrix.

    Returns:
        ( Tensor (..., d, d) ): Square root of `x`.
    """
    # code copied from
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    L, Q = torch.linalg.eigh(x)
    zero = torch.zeros((), dtype=L.dtype, device=L.device)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def dissipator(L: Tensor, rho: Tensor) -> Tensor:
    r"""Apply the Lindblad dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:

    $$
        \mathcal{D}[L](\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L ( Tensor (..., d, d) ): Jump operator (an arbitrary operator).
        rho ( Tensor (..., d, d) ): Density matrix.

    Returns:
        ( Tensor (..., d, d) ): Resulting density matrix.
    """
    return (
        L @ rho @ L.adjoint()
        - 0.5 * L.adjoint() @ L @ rho
        - 0.5 * rho @ L.adjoint() @ L
    )


def lindbladian(H: Tensor, Ls: Tensor, rho: Tensor) -> Tensor:
    r"""Apply the Lindbladian superoperator to a density matrix.

    The Lindbladian superoperator $\mathcal{L}$ is defined by:

    $$
        \mathcal{L}(\rho) = -i[H,\rho] + \sum_{k=1}^n \mathcal{D}[L_k](\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of jump operators (arbitrary
    operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator (see
    [dissipator()][torchqdynamics.dissipator]).

    Note:
        This superoperator is also sometimes called the *Liouvillian*.

    Args:
        H ( Tensor (..., d, d) ): Hamiltonian.
        Ls ( Tensor (..., n, d, d) ): Sequence of jump operators (arbitrary operators).
        rho ( Tensor (..., d, d) ): Density matrix.

    Returns:
        ( Tensor (..., d, d) ): Resulting operator (it is not a density matrix).
    """
    return -1j * (H @ rho - rho @ H) + dissipator(Ls, rho).sum(0)


def tensprod(*args: Tensor) -> Tensor:
    r"""Tensor product of a sequence of state vectors, density matrices or
    operators.

    Examples:
        >>> rho = tq.tensprod(
        ...     tq.coherent_dm(20, 2.0),
        ...     tq.fock_dm(2, 0),
        ...     tq.fock_dm(5, 1)
        ... )
        >>> rho.shape
        torch.Size([200, 200])

    The returned tensor has shape:

    - $(d, 1)$ with $d=\prod_k d_k$ if all input tensors are state vectors with shape
      $(d_k, 1)$,
    - $(d, d)$ with $d=\prod_k d_k$ if all input tensors are density matrices or
      operators vectors with shape $(d_k, d_k)$.

    Warning:
        This function does not yet support linear map.

    Warning:
        This function does not yet support arbitrarily batched tensors (see
        [issue #69](https://github.com/pierreguilmin/torchqdynamics/issues/69)).

    Note:
        This function is the equivalent of `qutip.tensor()`.

    Args:
        *args ( Sequence of Tensor (d_k, 1) or (d_k, d_k) ): Sequence of state vectors,
            density matrices or operators.

    Returns:
        ( Tensor (d, 1) or (d, d) ): Tensor product of the input tensors.
    """
    # TODO: adapt to bras
    return reduce(torch.kron, args)


def trace(x: Tensor) -> Tensor:
    """Trace of a matrix.

    Args:
        x ( Tensor (..., d, d) ): Matrix.

    Returns:
        ( Tensor (...) ): Trace of `x`.
    """
    return torch.einsum('...ii', x)


def ptrace(x: Tensor, keep: int | tuple[int, ...], dims: tuple[int, ...]) -> Tensor:
    """Partial trace of a state vector or density matrix.

    Examples:
        >>> rho = tq.tensprod(
        ...     tq.coherent_dm(20, 2.0),
        ...     tq.fock_dm(2, 0),
        ...     tq.fock_dm(5, 1)
        ... )
        >>> rhoA = tq.ptrace(rho, 0, (20, 2, 5))
        >>> rho1.shape
        torch.Size([20, 20])
        >>> rhoBC = tq.ptrace(rho, (1, 2), (20, 2, 5))
        >>> rhoBC.shape
        torch.Size([10, 10])

    Warning:
        This function does not yet support linear map.

    Args:
        x ( Tensor (..., d, 1) or (..., d, d) ): State vector or density matrix of a
            composite system.
        keep (int or tuple of ints): Dimensions to keep after partial trace.
        dims (tuple of ints): Dimensions of each subsystem in the composite system
            Hilbert space tensor product.

    Returns:
        ( Tensor (..., n, n) ): Resulting density matrix (with `n <= d`).
    """
    # TODO: adapt to bras
    # convert keep and dims to tensors
    keep = torch.as_tensor([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
    dims = torch.as_tensor(dims)  # e.g. [20, 2, 5]tq.
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
    batch_dims = x.shape[:-2]
    if is_ket(x):
        x = x.view(-1, *dims)  # e.g. (..., 20, 2, 5)
        eq = ''.join(['...'] + eq1 + [',...'] + eq2)  # e.g. '...abc,...ade'
        x = torch.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    else:
        x = x.view(-1, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = ''.join(['...'] + eq1 + eq2)  # e.g. '...abcade'
        x = torch.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)

    # reshape to final dimension
    nkeep = torch.prod(dims[keep])  # e.g. 10
    return x.reshape(*batch_dims, nkeep, nkeep)  # e.g. (..., 10, 10)


def expect(O: Tensor, x: Tensor) -> Tensor:
    r"""Expectation value of an operator on a state vector or a density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a state vector $\ket\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned tensor is complex-valued.

    Warning:
        This function does not yet support linear map.

    Args:
        O ( Tensor (d, d) ): Arbitrary operator.
        x ( Tensor (..., d, 1) or (..., d, d) ): State vector or density matrix.

    Returns:
        ( Tensor (...) ): Complex-valued expectation value.
    """
    # TODO: adapt to bras
    if is_ket(x):
        return torch.einsum('...ij,jk,...kl->...', x.adjoint(), O, x)  # <x|O|x>
    return torch.einsum('ij,...ji->...', O, x)  # tr(Ox)
