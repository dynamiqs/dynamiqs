from __future__ import annotations

from functools import reduce

import torch
from torch import Tensor

__all__ = [
    'trace',
    'expect',
    'norm',
    'unit',
    'tensprod',
    'ptrace',
    'dissipator',
    'lindbladian',
    'is_ket',
    'is_bra',
    'is_dm',
    'ket_to_bra',
    'ket_to_dm',
    'ket_overlap',
    'ket_fidelity',
    'dm_fidelity',
]


def trace(x: Tensor) -> Tensor:
    r"""Returns the trace of a tensor along its last two dimensions.

    Args:
        x _(..., n, n)_: Tensor.

    Returns:
        _(...)_ Trace of `x`.

    Examples:
        >>> x = torch.ones(3, 3)
        >>> dq.trace(x)
        tensor(3.)
        >>> # If the argument is batched, the trace is computed for each batch element
        >>> x = torch.stack([torch.ones(3, 3), torch.zeros(3, 3)])  # shape: (2, 3, 3)
        >>> dq.trace(x)                                             # shape: (2)
        tensor([3., 0.])
    """
    return torch.einsum('...ii', x)


def expect(O: Tensor, x: Tensor) -> Tensor:
    r"""Returns the expectation value of an operator on a ket, bra or density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a ket $\ket\psi$ or bra $\bra\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned tensor is complex-valued. If the operator $O$ corresponds to a
        physical observable, it is Hermitian: $O^\dag=O$, and the expectation value
        is real. One can then keep only the real values of the returned tensor using
        `dq.expect(O, x).real`.

    Args:
        O _(n, n)_: Arbitrary operator.
        x _(..., n, 1) or (..., 1, n) or (..., n, n)_: Ket, bra or density matrix.

    Returns:
        _(...)_ Complex-valued expectation value.

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.

    Examples:
        >>> a = dq.destroy(16)
        >>> n = a.mH @ a

        For a ket:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.expect(n, psi)
        tensor(4.000+0.j)

        For a density matrix:
        >>> rho = dq.coherent_dm(16, 1.0j)
        >>> dq.expect(n, rho)
        tensor(1.000+0.j)

        If the argument is batched, the expectation value is computed for each batch
        element:
        >>> fock0, fock1, fock2 = dq.fock(16, 0), dq.fock(16, 1), dq.fock(16, 2)
        >>> x = torch.stack([fock0, fock1, fock2])  # shape: (3, 16, 1)
        >>> dq.expect(n, x).real                    # shape: (3)
        tensor([0.000, 1.000, 2.000])
    """
    if is_ket(x):
        return torch.einsum('...ij,jk,...kl->...', x.mH, O, x)  # <x|O|x>
    elif is_bra(x):
        return torch.einsum('...ij,jk,...kl->...', x, O, x.mH)
    elif is_dm(x):
        return torch.einsum('ij,...ji->...', O, x)  # tr(Ox)
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {tuple(x.shape)}.'
        )


def norm(x: Tensor) -> Tensor:
    r"""Returns the norm of a ket, bra or density matrix.

    For a ket or a bra, the returned norm is $\sqrt{\braket{\psi|\psi}}$. For a density
    matrix, it is $\tr{\rho}$.

    Args:
        x _(..., n, 1) or (..., 1, n) or (..., n, n)_: Ket, bra or density matrix.

    Returns:
        _(...)_ Real-valued norm of `x`.

    Raises:
        ValueError: If `x`is not a ket, bra or density matrix.

    Examples:
        >>> import numpy as np

        For a ket:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(psi)
        tensor(1.414)

        For a density matrix:
        >>> rho = dq.fock_dm(4, 0) + dq.fock_dm(4, 1) + dq.fock_dm(4, 2)
        >>> dq.norm(rho)
        tensor(3.)

        If the argument is batched, the norm is computed for each batch element:
        >>> fock0, fock1 = dq.fock(8, 0), dq.fock(8, 1)
        >>> x = torch.stack([fock0, fock0 + fock1])  # shape: (2, 8, 1)
        >>> dq.norm(x)                               # shape: (2)
        tensor([1.000, 1.414])
    """
    if is_ket(x) or is_bra(x):
        return torch.linalg.norm(x, dim=(-2, -1)).real
    elif is_dm(x):
        return trace(x).real
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {tuple(x.shape)}.'
        )


def unit(x: Tensor) -> Tensor:
    r"""Normalize a ket, bra or density matrix to unit norm.

    The returned object is divided by its norm (see [norm()][dynamiqs.norm]).

    Args:
        x _(..., n, 1) or (..., 1, n) or (..., n, n)_: Ket, bra or density matrix.

    Returns:
        _(..., n, 1) or (..., 1, n) or (..., n, n)_ Normalized ket, bra or density
            matrix.

    Examples:
        For a ket:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> psi
        tensor([[1.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j]])
        >>> dq.norm(psi)
        tensor(1.414)
        >>> psi = dq.unit(psi)
        >>> psi
        tensor([[0.707+0.j],
                [0.707+0.j],
                [0.000+0.j],
                [0.000+0.j]])
        >>> dq.norm(psi)
        tensor(1.000)

        For a density matrix:
        >>> rho = dq.fock_dm(4, 0) + dq.fock_dm(4, 1) + dq.fock_dm(4, 2)
        >>> rho
        tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        >>> dq.norm(rho)
        tensor(3.)
        >>> rho = dq.unit(rho)
        >>> rho
        tensor([[0.333+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.333+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.333+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]])
        >>> dq.norm(rho)
        tensor(1.)

        If the argument is batched, each batch element is normalized separately:
        >>> fock0, fock1 = dq.fock(8, 0), dq.fock(8, 1)
        >>> x = torch.stack([fock0, fock0 + fock1])  # shape: (2, 8, 1)
        >>> dq.norm(x)
        tensor([1.000, 1.414])
        >>> x = dq.unit(x)                           # shape: (2, 8, 1)
        >>> dq.norm(x)
        tensor([1.000, 1.000])
    """
    return x / norm(x)[..., None, None]


def tensprod(*args: Tensor) -> Tensor:
    r"""Returns the tensor product of multiple kets, bras, density matrices or
    operators.

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
        *args _(..., n_k, 1) or (..., 1, n_k) or (..., n_k, n_k)_: Variable length
            argument list of kets, density matrices or operators.

    Returns:
        _(..., n, 1) or (..., 1, n) or (..., n, n)_ Tensor product of the input tensors.

    Examples:
        For a ket:
        >>> psi = dq.tensprod(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi.shape
        torch.Size([60, 1])

        For a density matrix:
        >>> rho = dq.tensprod(dq.fock_dm(3, 0), dq.fock_dm(4, 2), dq.fock_dm(5, 1))
        >>> rho.shape
        torch.Size([60, 60])

        If the arguments are batched, the tensor product is computed for each batch
        element:
        >>> x = torch.stack([dq.fock(3, 0), dq.fock(3, 1)])  # shape: (2, 3, 1)
        >>> y = torch.stack([dq.fock(4, 0), dq.fock(4, 1)])  # shape: (2, 4, 1)
        >>> xy = dq.tensprod(x, y)                           # shape: (2, 12, 1)
        >>> all(xy[0] == dq.tensprod(x[0], y[0]))
        True
        >>> all(xy[1] == dq.tensprod(x[1], y[1]))
        True
    """
    return reduce(_bkron, args)


def _bkron(x: Tensor, y: Tensor) -> Tensor:
    """Returns the batched Kronecker product of two matrices."""
    x_type = _quantum_type(x)
    y_type = _quantum_type(y)
    if x_type != y_type:
        raise ValueError(
            'Arguments `x` and `y` have incompatible quantum types for tensor product:'
            f' `x` is a {x_type} with shape {tuple(x.shape)}, but  `y` is a {y_type}'
            f' with shape {tuple(y.shape)}.'
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


def ptrace(x: Tensor, keep: int | tuple[int, ...], dims: tuple[int, ...]) -> Tensor:
    r"""Returns the partial trace of a ket, bra or density matrix.

    Args:
        x _(..., n, 1) or (..., 1, n) or (..., n, n)_: Ket, bra or density matrix of a
            composite system.
        keep _(int or tuple of ints)_: Dimensions to keep after partial trace.
        dims _(tuple of ints)_: Dimensions of each subsystem in the composite system
            Hilbert space tensor product.

    Returns:
        _(..., m, m)_ Density matrix (with `m <= n`).

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.
        ValueError: If `dims` does not match the shape of `x`, or if `keep` is
            incompatible with `dims`.

    Note:
        The returned object is always a density matrix, even if the input is a ket or a
        bra.

    Examples:
        For a ket:
        >>> psi_abc = dq.tensprod(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi_abc.shape
        torch.Size([60, 1])
        >>> rho_a = dq.ptrace(psi_abc, 0, (3, 4, 5))
        >>> rho_a.shape
        torch.Size([3, 3])
        >>> rho_bc = dq.ptrace(psi_abc, (1, 2), (3, 4, 5))
        >>> rho_bc.shape
        torch.Size([20, 20])

        For a density matrix:
        >>> rho_abc = dq.tensprod(dq.fock_dm(3, 0), dq.fock_dm(4, 2), dq.fock_dm(5, 1))
        >>> rho_abc.shape
        torch.Size([60, 60])
        >>> rho_a = dq.ptrace(rho_abc, 0, (3, 4, 5))
        >>> rho_a.shape
        torch.Size([3, 3])
        >>> rho_bc = dq.ptrace(rho_abc, (1, 2), (3, 4, 5))
        >>> rho_bc.shape
        torch.Size([20, 20])

        If the argument is batched, the partial trace is computed for each batch
        element:
        >>> psi1 = dq.tensprod(dq.fock(3, 0), dq.fock(4, 1), dq.fock(5, 1))
        >>> psi2 = dq.tensprod(dq.fock(3, 1), dq.fock(4, 3), dq.fock(5, 2))
        >>> x_abc = torch.stack([psi1, psi2])     # shape: (2, 60, 1)
        >>> x_a = dq.ptrace(x_abc, 0, (3, 4, 5))  # shape: (2, 3, 3)
        >>> torch.all(x_a[0] == dq.ptrace(x_abc[0], 0, (3, 4, 5)))
        tensor(True)
        >>> torch.all(x_a[1] == dq.ptrace(x_abc[1], 0, (3, 4, 5)))
        tensor(True)
    """
    # convert keep and dims to tensors
    keep = torch.as_tensor([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
    dims = torch.as_tensor(dims)  # e.g. [20, 2, 5]
    ndims = len(dims)  # e.g. 3

    # check that input dimensions match
    hilbert_size = x.size(-2) if is_ket(x) else x.size(-1)
    prod_dims = torch.prod(dims)
    if not prod_dims == hilbert_size:
        dims_prod_str = '*'.join(str(d.item()) for d in dims) + f'={prod_dims}'
        raise ValueError(
            'Argument `dims` must match the Hilbert space dimension of `x` of'
            f' {hilbert_size}, but the product of its values is {dims_prod_str}.'
        )
    if torch.any(keep < 0) or torch.any(keep > len(dims) - 1):
        raise ValueError(
            'Argument `keep` must match the Hilbert space structure specified by'
            ' `dims`.'
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
    if is_ket(x) or is_bra(x):
        x = x.view(-1, *dims)  # e.g. (..., 20, 2, 5)
        eq = ''.join(['...'] + eq1 + [',...'] + eq2)  # e.g. '...abc,...ade'
        x = torch.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    elif is_dm(x):
        x = x.view(-1, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = ''.join(['...'] + eq1 + eq2)  # e.g. '...abcade'
        x = torch.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {tuple(x.shape)}.'
        )

    # reshape to final dimension
    nkeep = torch.prod(dims[keep])  # e.g. 10
    return x.reshape(*batch_dims, nkeep, nkeep)  # e.g. (..., 10, 10)


def dissipator(L: Tensor, rho: Tensor) -> Tensor:
    r"""Applies the Lindblad dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:

    $$
        \mathcal{D}[L](\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L _(..., n, n)_: Jump operator (an arbitrary operator).
        rho _(..., n, n)_: Density matrix.

    Returns:
        _(..., n, n)_ Density matrix.

    Examples:
        >>> L = dq.destroy(4)
        >>> rho = dq.fock_dm(4, 2)
        >>> dq.dissipator(L, rho)
        tensor([[ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j],
                [ 0.000+0.j,  2.000+0.j,  0.000+0.j,  0.000+0.j],
                [ 0.000+0.j,  0.000+0.j, -2.000+0.j,  0.000+0.j],
                [ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j]])

        If the arguments are batched, the dissipation superoperator is computed for
        each batch element:
        >>> L = torch.stack([dq.destroy(4), dq.create(4)])         # shape: (2, 4, 4)
        >>> x = torch.stack([dq.fock_dm(4, 2), dq.fock_dm(4, 2)])  # shape: (2, 4, 4)
        >>> dq.dissipator(L, x)                                    # shape: (2, 4, 4)
        tensor([[[ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  2.000+0.j,  0.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.000+0.j, -2.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j]],
        <BLANKLINE>
                [[ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.000+0.j,  0.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.000+0.j, -3.000+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.000+0.j,  0.000+0.j,  3.000+0.j]]])
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
        H _(..., n, n)_: Hamiltonian.
        Ls _(..., N, n, n)_: Sequence of jump operators (arbitrary operators).
        rho _(..., n, n)_: Density matrix.

    Returns:
        _(..., n, n)_ Resulting operator (it is not a density matrix).

    Examples:
        For example for a lossy quantum harmonic oscillator with Hamiltonian
        $H=\omega a^\dag a$ and two jump operators $L_1=\sqrt{\kappa_1} a$ and
        $L_\varphi=\sqrt{2\kappa_\varphi} a^\dag a$:
        >>> import numpy as np
        >>> a = dq.destroy(4)
        >>> delta = 1.0
        >>> kappa_1 = 0.1
        >>> kappa_phi = 0.05
        >>> H = delta * a.mH @ a
        >>> L1 = np.sqrt(kappa_1) * a
        >>> Lphi = np.sqrt(2 * kappa_phi) * a.mH @ a
        >>> Ls = torch.stack([L1, Lphi])
        >>> rho = dq.coherent_dm(4, 0.5)
        >>> dq.lindbladian(H, Ls, rho)
        tensor([[ 0.019+0.000j, -0.029+0.389j, -0.038+0.275j, -0.025+0.125j],
                [-0.029-0.389j, -0.015+0.000j, -0.012+0.069j, -0.008+0.042j],
                [-0.038-0.275j, -0.012-0.069j, -0.004+0.000j, -0.002+0.007j],
                [-0.025-0.125j, -0.008-0.042j, -0.002-0.007j, -0.001+0.000j]])

        If the arguments are batched, the Lindbladian is computed for each batch
        element (see [dissipator()][dynamiqs.dissipator] for an example with batching).
    """
    return -1j * (H @ rho - rho @ H) + dissipator(Ls, rho).sum(0)


def is_ket(x: Tensor) -> bool:
    r"""Returns True if a tensor is in the format of a ket.

    Args:
        x _(...)_: Tensor.

    Returns:
        True if the last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.is_ket(torch.ones(3, 1))
        True
        >>> dq.is_ket(torch.ones(5, 3, 1))
        True
        >>> dq.is_ket(torch.ones(3, 3))
        False
    """
    return x.size(-1) == 1


def is_bra(x: Tensor) -> bool:
    r"""Returns True if a tensor is in the format of a bra.

    Args:
        x _(...)_: Tensor.

    Returns:
        True if the second to last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.is_bra(torch.ones(1, 3))
        True
        >>> dq.is_bra(torch.ones(5, 1, 3))
        True
        >>> dq.is_bra(torch.ones(3, 3))
        False
    """
    return x.size(-2) == 1


def is_dm(x: Tensor) -> bool:
    r"""Returns True if a tensor is in the format of a density matrix.

    Args:
        x _(...)_: Tensor.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.

    Examples:
        >>> dq.is_dm(torch.ones(3, 3))
        True
        >>> dq.is_dm(torch.ones(5, 3, 3))
        True
        >>> dq.is_dm(torch.ones(3, 1))
        False
    """
    return x.size(-1) == x.size(-2)


def _quantum_type(x: Tensor) -> str:
    """Returns the quantum type of a tensor."""
    if is_ket(x):
        return 'ket'
    elif is_bra(x):
        return 'bra'
    elif is_dm(x):
        return 'density matrix'
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {tuple(x.shape)}.'
        )


def ket_to_bra(x: Tensor) -> Tensor:
    r"""Returns the bra $\bra\psi$ associated to a ket $\ket\psi$.

    Args:
        x _(..., n, 1)_: Ket.

    Returns:
        _(..., 1, n)_ Bra.

    Examples:
        >>> psi = dq.fock(3, 1)  # shape: (3, 1)
        >>> psi
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.ket_to_bra(psi)  # shape: (1, 3)
        tensor([[0.-0.j, 1.-0.j, 0.-0.j]])

        If the argument is batched, the associated bra is computed for each batch
        element:
        >>> x = torch.stack([dq.fock(3, 0), dq.fock(3, 1)])  # shape: (2, 3, 1)
        >>> x
        tensor([[[1.+0.j],
                 [0.+0.j],
                 [0.+0.j]],
        <BLANKLINE>
                [[0.+0.j],
                 [1.+0.j],
                 [0.+0.j]]])
        >>> dq.ket_to_bra(x)                                 # shape: (2, 1, 3)
        tensor([[[1.-0.j, 0.-0.j, 0.-0.j]],
        <BLANKLINE>
                [[0.-0.j, 1.-0.j, 0.-0.j]]])
    """
    return x.mH


def ket_to_dm(x: Tensor) -> Tensor:
    r"""Returns the density matrix $\ket\psi\bra\psi$ associated to a ket $\ket\psi$.

    Args:
        x _(..., n, 1)_: Ket.

    Returns:
        _(..., n, n)_ Density matrix.

    Examples:
        >>> psi = dq.fock(3, 1)  # shape: (3, 1)
        >>> psi
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.ket_to_dm(psi)  # shape: (3, 3)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]])

        If the argument is batched, the associated density matrix is computed for each
        batch element:
        >>> x = torch.stack([dq.fock(3, 0), dq.fock(3, 1)])  # shape: (2, 3, 1)
        >>> x
        tensor([[[1.+0.j],
                 [0.+0.j],
                 [0.+0.j]],
        <BLANKLINE>
                [[0.+0.j],
                 [1.+0.j],
                 [0.+0.j]]])
        >>> dq.ket_to_dm(x)                                  # shape: (2, 3, 3)
        tensor([[[1.+0.j, 0.+0.j, 0.+0.j],
                 [0.+0.j, 0.+0.j, 0.+0.j],
                 [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
                [[0.+0.j, 0.+0.j, 0.+0.j],
                 [0.+0.j, 1.+0.j, 0.+0.j],
                 [0.+0.j, 0.+0.j, 0.+0.j]]])
    """
    return x @ ket_to_bra(x)


def ket_overlap(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the overlap (inner product) $\braket{\varphi,\psi}$ between two kets
    $\ket\varphi$ and $\ket\psi$.

    Args:
        x _(..., n, 1)_: First ket.
        y _(..., n, 1)_: Second ket.

    Returns:
        _(...)_ Complex-valued overlap.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.ket_overlap(fock0, fock0)
        tensor(1.+0.j)
        >>> fock1 = dq.fock(3, 1)
        >>> dq.ket_overlap(fock0, fock1)
        tensor(0.+0.j)
        >>> coh0 = dq.coherent(8, 0.0)
        >>> coh1 = dq.coherent(8, 1.0)
        >>> dq.ket_overlap(coh0, coh1)
        tensor(0.607+0.j)

        If the arguments are batched, the overlap is computed for each batch element:
        >>> x = torch.stack([dq.fock(3, 0), dq.fock(3, 1)])  # shape: (2, 3, 1)
        >>> y = torch.stack([dq.fock(3, 0), dq.fock(3, 2)])  # shape: (2, 3, 1)
        >>> dq.ket_overlap(x, y)                             # shape: (2)
        tensor([1.+0.j, 0.+0.j])
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
        x _(..., n, 1)_: First ket.
        y _(..., n, 1)_: Second ket.

    Returns:
        _(...)_ Real-valued fidelity.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.ket_fidelity(fock0, fock0)
        tensor(1.)
        >>> fock1 = dq.fock(3, 1)
        >>> dq.ket_fidelity(fock0, fock1)
        tensor(0.)
        >>> coh0 = dq.coherent(8, 0.0)
        >>> coh1 = dq.coherent(8, 1.0)
        >>> dq.ket_fidelity(coh0, coh1)
        tensor(0.368)

        If the arguments are batched, the fidelity is computed for each batch element:
        >>> x = torch.stack([dq.fock(3, 0), dq.fock(3, 1)])  # shape: (2, 3, 1)
        >>> y = torch.stack([dq.fock(3, 0), dq.fock(3, 2)])  # shape: (2, 3, 1)
        >>> dq.ket_fidelity(x, y)                            # shape: (2)
        tensor([1., 0.])
    """
    return ket_overlap(x, y).abs().pow(2).real


def dm_fidelity(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the fidelity of two density matrices.

    The fidelity of two density matrices $\rho$ and $\sigma$ is defined by:

    $$
        F(\rho, \sigma) = \tr{\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}}^2.
    $$

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Args:
        x _(..., n, n)_: First density matrix.
        y _(..., n, n)_: Second density matrix.

    Returns:
        _(...)_ Real-valued fidelity.

    Examples:
        >>> fock0 = dq.fock_dm(3, 0)
        >>> dq.dm_fidelity(fock0, fock0)
        tensor(1.)
        >>> fock1 = dq.fock_dm(3, 1)
        >>> dq.dm_fidelity(fock0, fock1)
        tensor(0.)
        >>> coh0 = dq.coherent_dm(8, 0.0)
        >>> coh1 = dq.coherent_dm(8, 1.0)
        >>> dq.dm_fidelity(coh0, coh1)
        tensor(0.368)

        If the arguments are batched, the fidelity is computed for each batch element:
        >>> x = torch.stack([dq.fock_dm(3, 0), dq.fock_dm(3, 1)])  # shape: (2, 3, 3)
        >>> y = torch.stack([dq.fock_dm(3, 0), dq.fock_dm(3, 2)])  # shape: (2, 3, 3)
        >>> dq.dm_fidelity(x, y)                                   # shape: (2)
        tensor([1., 0.])
    """
    sqrtm_x = _sqrtm(x)
    tmp = sqrtm_x @ y @ sqrtm_x

    # we don't need the whole matrix `sqrtm(tmp)`, just its trace, which can be computed
    # by summing the square roots of `tmp` eigenvalues
    eigvals_tmp = torch.linalg.eigvalsh(tmp)

    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    eigvals_tmp = eigvals_tmp.where(eigvals_tmp >= 0, zero)

    trace_sqrtm_tmp = torch.sqrt(eigvals_tmp).sum(-1)

    return trace_sqrtm_tmp.pow(2).real


def _sqrtm(x: Tensor) -> Tensor:
    """Returns the square root of a symmetric or Hermitian positive definite matrix.

    Args:
        x _(..., n, n)_: Symmetric or Hermitian positive definite matrix.

    Returns:
        _(..., n, n)_ Square root of `x`.
    """
    # code copied from
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    L, Q = torch.linalg.eigh(x)
    zero = torch.zeros((), dtype=L.dtype, device=L.device)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH
