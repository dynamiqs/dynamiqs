from math import sqrt

import torch
from torch import Tensor

from .operators import eye

__all__ = [
    'operator_to_vector',
    'vector_to_operator',
    'spre',
    'spost',
    'sprepost',
    'sdissipator',
    'slindbladian',
]


def operator_to_vector(x: Tensor) -> Tensor:
    r"""Returns the vectorized version of an operator.

    Args:
        x _(..., n, n)_: Operator.

    Returns:
        _(..., n^2, 1)_ Vectorized operator.

    Examples:
        >>> dq.destroy(3)
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j]])
        >>> dq.operator_to_vector(dq.destroy(3))
        tensor([[0.000+0.j],
                [1.000+0.j],
                [0.000+0.j],
                [0.000+0.j],
                [0.000+0.j],
                [1.414+0.j],
                [0.000+0.j],
                [0.000+0.j],
                [0.000+0.j]])
    """
    batch_sizes = x.shape[:-2]
    return x.view(*batch_sizes, -1, 1)


def vector_to_operator(x: Tensor) -> Tensor:
    r"""Returns the operator version of a vectorized operator.

    Args:
        x _(..., n^2, 1)_: Vectorized operator.

    Returns:
        _(..., n, n)_ Operator.
    """
    batch_sizes = x.shape[:-2]
    n = int(sqrt(x.shape[-2]))
    return x.view(*batch_sizes, n, n)


def spre(x: Tensor) -> Tensor:
    r"""Returns the superoperator formed from pre-multiplication by an operator.

    Args:
        x _(..., n, n)_: Operator.

    Returns:
        _(..., n^2, n^2)_ Pre-multiplication superoperator.

    Examples:
        >>> dq.spre(dq.destroy(3)).shape
        torch.Size([9, 9])
    """
    n = x.size(-1)
    I = eye(n)
    return torch.kron(I, x)


def spost(x: Tensor) -> Tensor:
    r"""Returns the superoperator formed from post-multiplication by an operator.

    Args:
        x _(..., n, n)_: Operator.

    Returns:
        _(..., n^2, n^2)_ Post-multiplication superoperator.

    Examples:
        >>> dq.spost(dq.destroy(3)).shape
        torch.Size([9, 9])
    """
    n = x.size(-1)
    I = eye(n)
    # torch.kron has undocumented dependence on memory layout, so we need to use
    # contiguous() here, see e.g. https://github.com/pytorch/pytorch/issues/74442
    # and https://github.com/pytorch/pytorch/issues/54135
    return torch.kron(x.mT.contiguous(), I)


def _bkron(A: Tensor, B: Tensor) -> Tensor:
    """Compute the batched Kronecker product of two matrices A and B."""
    # extracting shapes
    A_shape = A.shape
    B_shape = B.shape

    # ensuring A and B are batched in the same way
    assert A_shape[:-2] == B_shape[:-2], 'The batches of A and B must be the same size'

    # reshaping A and B for the computation
    reshaped_A = A.unsqueeze(-1).unsqueeze(-3)
    reshaped_B = B.unsqueeze(-2).unsqueeze(-4)

    # computing the product
    result = reshaped_A * reshaped_B

    # reshaping to the final desired shape (..., n*m, n*m)
    return result.reshape(
        *A_shape[:-2], A_shape[-2] * B_shape[-2], A_shape[-1] * B_shape[-1]
    )


def sprepost(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the superoperator formed from pre- and post-multiplication by operators.

    Args:
        x _(..., n, n)_: Operator for pre-multiplication.
        y _(..., n, n)_: Operator for post-multiplication.

    Returns:
        _(..., n^2, n^2)_ Pre- and post-multiplication superoperator.

    Examples:
        >>> dq.sprepost(dq.destroy(3), dq.create(3)).shape
        torch.Size([9, 9])
    """
    return _bkron(y.mT, x)


def sdissipator(L: Tensor) -> Tensor:
    r"""Returns the Lindblad dissipation superoperator (in matrix form).

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:
    $$
        \mathcal{D}[L] (\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L _(..., n, n)_: Jump operator (an arbitrary operator).

    Returns:
        _(..., n^2, n^2)_ Dissipator superoperator.
    """
    LdagL = L.mH @ L
    return sprepost(L, L.mH) - 0.5 * spre(LdagL) - 0.5 * spost(LdagL)


def slindbladian(H: Tensor, L: Tensor) -> Tensor:
    r"""Returns the Lindbladian superoperator (in matrix form).

    The Lindbladian superoperator $\mathcal{L}$ is defined by:
    $$
        \mathcal{L} (\rho) = -i[H,\rho] + \sum_{k=1}^N \mathcal{D}[L_k] (\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of $N$ jump operators
    (arbitrary operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator
    (see [sdissipator()][dynamiqs.sdissipator]).

    Notes:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H _(..., n, n)_: Hamiltonian.
        L _(..., N, n, n)_: Sequence of jump operators (arbitrary operators).

    Returns:
        _(..., n^2, n^2)_ Lindbladian superoperator.
    """
    return 1j * (spre(H) - spost(H)) + sdissipator(L).sum(-3)
