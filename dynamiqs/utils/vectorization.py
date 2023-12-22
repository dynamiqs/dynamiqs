from math import sqrt

import torch
from torch import Tensor

from .operators import eye
from .utils import _bkron

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

    The vectorized column vector $\kett{A}$ (shape $n^2\times 1$) is obtained by
    stacking the columns of the matrix $A$ (shape $n\times n$) on top of one another:
    $$
        A = \begin{pmatrix} a & b \\\\ c & d \end{pmatrix}
        \to
        \kett{A} = \begin{pmatrix} a \\\\ c \\\\ b \\\\ d \end{pmatrix}.
    $$

    Args:
        x _(..., n, n)_: Operator.

    Returns:
        _(..., n^2, 1)_ Vectorized operator.

    Examples:
        >>> A = torch.tensor([[1, 2], [3, 4]])
        >>> A
        tensor([[1, 2],
                [3, 4]])
        >>> dq.operator_to_vector(A)
        tensor([[1],
                [3],
                [2],
                [4]])
    """
    batch_sizes = x.shape[:-2]
    return x.mT.reshape(*batch_sizes, -1, 1)


def vector_to_operator(x: Tensor) -> Tensor:
    r"""Returns the operator version of a vectorized operator.

    The matrix $A$ (shape $n\times n$) is obtained by stacking horizontally next to
    each other each group of $n$ elements of the vectorized column vector $\kett{A}$
    (shape $n^2\times 1$):
    $$
        \kett{A} = \begin{pmatrix} a \\\\ b \\\\ c \\\\ d \end{pmatrix}
        \to
        A = \begin{pmatrix} a & c \\\\ b & d \end{pmatrix}.
    $$

    Args:
        x _(..., n^2, 1)_: Vectorized operator.

    Returns:
        _(..., n, n)_ Operator.

    Examples:
        >>> Avec = torch.tensor([[1], [2], [3], [4]])
        >>> Avec
        tensor([[1],
                [2],
                [3],
                [4]])
        >>> dq.vector_to_operator(Avec)
        tensor([[1, 3],
                [2, 4]])
    """
    batch_sizes = x.shape[:-2]
    n = int(sqrt(x.shape[-2]))
    return x.reshape(*batch_sizes, n, n).mT


def spre(x: Tensor) -> Tensor:
    r"""Returns the superoperator formed from pre-multiplication by an operator.

    Pre-multiplication by matrix $A$ is defined by the superoperator
    $I_n \otimes A$ in vectorized form:
    $$
        AX \to (I_n \otimes A) \kett{X}.
    $$

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

    Post-multiplication by matrix $A$ is defined by the superoperator
    $A^\mathrm{T} \otimes I_n$ in vectorized form:
    $$
        XA \to (A^\mathrm{T} \otimes I_n) \kett{X}.
    $$

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


def sprepost(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the superoperator formed from pre- and post-multiplication by operators.

    Pre-multiplication by matrix $A$ and post-multiplication by matrix $B$ is defined
    by the superoperator $B^\mathrm{T} \otimes A$ in vectorized form:
    $$
        AXB \to (B^\mathrm{T} \otimes A) \kett{X}.
    $$

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

    The vectorized form of this superoperator is:
    $$
        L^\* \otimes L
        - \frac{1}{2} (I_n \otimes L^\dag L)
        - \frac{1}{2} (L^\mathrm{T} L^\* \otimes I_n).
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
    (see [`dq.sdissipator()`][dynamiqs.sdissipator]).

    The vectorized form of this superoperator is:
    $$
        -i (I_n \otimes H) + i (H^\mathrm{T} \otimes I_n) + \sum_{k=1}^N \left(
            L_k^\* \otimes L_k
            - \frac{1}{2} (I_n \otimes L_k^\dag L_k)
            - \frac{1}{2} (L_k^\mathrm{T} L_k^\* \otimes I_n)
        \right).
    $$

    Notes:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H _(..., n, n)_: Hamiltonian.
        L _(..., N, n, n)_: Sequence of jump operators (arbitrary operators).

    Returns:
        _(..., n^2, n^2)_ Lindbladian superoperator.
    """
    return -1j * (spre(H) - spost(H)) + sdissipator(L).sum(0)
