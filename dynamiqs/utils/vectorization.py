from math import sqrt

import torch
from torch import Tensor

from .operators import eye

__all__ = [
    'operator_to_vector',
    'vector_to_operator',
    'spre',
    'spost',
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
    return torch.kron(x, I)


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
    return torch.kron(I, x.mT.contiguous())


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
    return spre(L) @ spost(L.mH) - 0.5 * spre(LdagL) - 0.5 * spost(LdagL)


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
    return -1j * (spre(H) - spost(H)) + sdissipator(L).sum(-3)
