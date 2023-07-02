from __future__ import annotations

from functools import reduce

import torch
from torch import Tensor

__all__ = [
    'is_ket',
    'is_bra',
    'is_dm',
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
    """Returns True if a tensor is in the format of a ket.

    Args:
        x (...): Tensor.

    Returns:
        True if the last dimension of `x` is 1, False otherwise.
    """
    return x.size(-1) == 1


def is_bra(x: Tensor) -> bool:
    """Returns True if a tensor is in the format of a bra.

    Args:
        x (...): Tensor.

    Returns:
        True if the second to last dimension of `x` is 1, False otherwise.
    """
    return x.size(-2) == 1


def is_dm(x: Tensor) -> bool:
    """Returns True if a tensor is in the format of a density matrix.

    Args:
        x (...): Tensor.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.
    """
    return x.size(-1) == x.size(-2)


def _quantum_type(x: Tensor) -> str:
    """Returns the quantum type of a tensor."""
    if is_ket(x):
        return 'ket'
    elif is_bra(x):
        return 'bra'
    elif is_dm(x):
        return 'dm'
    else:
        raise ValueError('Tensor is not a ket, bra or density matrix.')


def ket_to_bra(x: Tensor) -> Tensor:
    r"""Returns the bra $\bra\psi$ associated to a ket $\ket\psi$.

    Args:
        x (..., n, 1): Ket.

    Returns:
        (..., 1, n): Bra.
    """
    return x.mH


def ket_to_dm(x: Tensor) -> Tensor:
    r"""Returns the density matrix $\ket\psi\bra\psi$ associated to a ket $\ket\psi$.

    Args:
        x (..., n, 1): Ket.

    Returns:
        (..., n, n): Density matrix.
    """
    return x @ ket_to_bra(x)


def ket_overlap(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the overlap (inner product) $\braket{\varphi,\psi}$ between two kets
    $\ket\varphi$ and $\ket\psi$.

    Args:
        x (..., n, 1): First ket.
        y (..., n, 1): Second ket.

    Returns:
        (...): Complex-valued overlap.
    """
    return (ket_to_bra(x) @ y).squeeze(-1).sum(-1)


def ket_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the fidelity of two kets.

    The fidelity of two pure states $\ket\varphi$ and $\ket\psi$ is defined by their
    squared overlap:

    $$
        F(\ket\varphi, \ket\psi) = \left|\braket{\varphi, \psi}\right|^2.
    $$

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Args:
        x (..., n, 1): First ket.
        y (..., n, 1): Second ket.

    Returns:
        (...): Real-valued fidelity.
    """
    return ket_overlap(x, y).abs().pow(2).real


def dm_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the fidelity of two density matrices.

    The fidelity of two density matrices $\rho$ and $\sigma$ is defined by:

    $$
        F(\rho, \sigma) = \tr{\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}}^2
    $$

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Args:
        x (..., n, n): First density matrix.
        y (..., n, n): Second density matrix.

    Returns:
        (...): Real-valued fidelity.
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
    """Returns the square root of a symmetric or Hermitian positive definite matrix.

    Args:
        x (..., n, n): Symmetric or Hermitian positive definite matrix.

    Returns:
        (..., n, n): Square root of `x`.
    """
    # code copied from
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    L, Q = torch.linalg.eigh(x)
    zero = torch.zeros((), dtype=L.dtype, device=L.device)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def dissipator(L: Tensor, rho: Tensor) -> Tensor:
    r"""Applies the Lindblad dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:

    $$
        \mathcal{D}[L](\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L (..., n, n): Jump operator (an arbitrary operator).
        rho (..., n, n): Density matrix.

    Returns:
        (..., n, n): Density matrix.
    """
    return L @ rho @ L.mH - 0.5 * L.mH @ L @ rho - 0.5 * rho @ L.mH @ L


def lindbladian(H: Tensor, Ls: Tensor, rho: Tensor) -> Tensor:
    r"""Applies the Lindbladian superoperator to a density matrix.

    The Lindbladian superoperator $\mathcal{L}$ is defined by:

    $$
        \mathcal{L}(\rho) = -i[H,\rho] + \sum_{k=1}^N \mathcal{D}[L_k](\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of $N$ jump operators
    (arbitrary operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator
    (see [dissipator()][dynamiqs.dissipator]).

    Note:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H (..., n, n): Hamiltonian.
        Ls (..., N, n, n): Sequence of jump operators (arbitrary operators).
        rho (..., n, n): Density matrix.

    Returns:
        (..., n, n): Resulting operator (it is not a density matrix).
    """
    return -1j * (H @ rho - rho @ H) + dissipator(Ls, rho).sum(0)


def tensprod(*args: Tensor) -> Tensor:
    r"""Returns the tensor product of a sequence of kets, bras, density matrices or
    operators.

    Examples:
        >>> psi = dq.tensprod(
        ...     dq.coherent(20, 2.0),
        ...     dq.fock(2, 0),
        ...     dq.fock(5, 1)
        ... )
        >>> psi.shape
        torch.Size([200, 1])

        >>> rho = dq.tensprod(
        ...     dq.coherent_dm(20, 2.0),
        ...     dq.fock_dm(2, 0),
        ...     dq.fock_dm(5, 1)
        ... )
        >>> rho.shape
        torch.Size([200, 200])

    The returned tensor shape is:

    - $(..., n, 1)$ with $n=\prod_k n_k$ if all input tensors are kets with shape
      $(..., n_k, 1)$,
    - $(..., 1, n)$ with $n=\prod_k n_k$ if all input tensors are bras with shape
      $(..., 1, n_k)$,
    - $(..., n, n)$ with $n=\prod_k n_k$ if all input tensors are density matrices or
      operators vectors with shape $(..., n_k, n_k)$.

    Note:
        This function is the equivalent of `qutip.tensor()`.

    Args:
        *args (..., n_k, 1) or (..., 1, n_k) or (..., n_k, n_k): Sequence of kets,
            density matrices or operators.

    Returns:
        (..., n, 1) or (..., 1, n) or (..., n, n): Tensor product of the input tensors.
    """
    return reduce(_bkron, args)


def _bkron(x: Tensor, y: Tensor) -> Tensor:
    """Returns the batched Kronecker product of two matrices."""
    x_type = _quantum_type(x)
    y_type = _quantum_type(y)
    if x_type != y_type:
        raise TypeError(
            'Arguments have incompatible quantum types for tensor product (`x` is a'
            f' {x_type} with shape {x.size()} and `y` is a {y_type} with shape'
            f' {y.size()}).'
        )

    # x: (..., x1, x2)
    # y: (..., y1, y2)

    batch_dims = x.shape[:-2]
    x1, x2 = x.shape[-2:]
    y1, y2 = y.shape[-2:]
    kron_dims = torch.Size((x1 * y1, x2 * y2))

    # perform element-wise multiplication of appropriately unsqueezed tensors to
    # simulate the Kronecker product
    x_tmp = x.unsqueeze(-1).unsqueeze(-3)  # (..., x1, 1, x2, 1)
    y_tmp = y.unsqueeze(-2).unsqueeze(-4)  # (..., 1, y1, 1, y2)
    out = x_tmp * y_tmp  # (..., x1, y1, x2, y2)

    # reshape the output
    return out.reshape(batch_dims + kron_dims)  # (..., x1 * y1, x2 * y2)


def trace(x: Tensor) -> Tensor:
    """Returns the trace of a tensor along its last two dimensions.

    Args:
        x (..., n, n): Tensor.

    Returns:
        (...): Trace of `x`.
    """
    return torch.einsum('...ii', x)


def ptrace(x: Tensor, keep: int | tuple[int, ...], dims: tuple[int, ...]) -> Tensor:
    """Returns the partial trace of a ket or density matrix.

    Examples:
        >>> rhoABC = dq.tensprod(
        ...     dq.coherent_dm(20, 2.0),
        ...     dq.fock_dm(2, 0),
        ...     dq.fock_dm(5, 1)
        ... )
        >>> rhoABC.shape
        torch.Size([200, 200])
        >>> rhoA = dq.ptrace(rho, 0, (20, 2, 5))
        >>> rhoA.shape
        torch.Size([20, 20])
        >>> rhoBC = dq.ptrace(rho, (1, 2), (20, 2, 5))
        >>> rhoBC.shape
        torch.Size([10, 10])

    Warning:
        This function does not yet support tensors in the bra format.

    Args:
        x (..., n, 1) or (..., n, n): Ket or density matrix of a composite system.
        keep (int or tuple of ints): Dimensions to keep after partial trace.
        dims (tuple of ints): Dimensions of each subsystem in the composite system
            Hilbert space tensor product.

    Returns:
        (..., m, m): Density matrix (with `m <= n`).

    Raises:
        ValueError: If `dims` does not match the input tensor shape, or if `keep` is incompatible with `dims`.
    """
    # TODO: adapt to bras
    # convert keep and dims to tensors
    keep = torch.as_tensor([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
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
    r"""Returns the expectation value of an operator on a ket or a density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a ket $\ket\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned tensor is complex-valued. If the operator $O$ corresponds to a
        physical observable, it is Hermitian: $O^\dag=O$, and the expectation value
        is real. One can then keep only the real values of the returned tensor using
        `dq.expect(O, x).real`.

    Warning:
        This function does not yet support tensors in the bra format.

    Args:
        O (n, n): Arbitrary operator.
        x (..., n, 1) or (..., n, n): Ket or density matrix.

    Returns:
        (...): Complex-valued expectation value.
    """
    # TODO: adapt to bras
    if is_ket(x):
        return torch.einsum('...ij,jk,...kl->...', x.mH, O, x)  # <x|O|x>
    return torch.einsum('ij,...ji->...', O, x)  # tr(Ox)
