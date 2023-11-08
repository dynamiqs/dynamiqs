from __future__ import annotations

from math import prod
from typing import get_args

import torch
from torch import Tensor

from .operators import displace
from .tensor_types import ArrayLike, Number, get_cdtype, to_tensor
from .utils import tensprod, todm

__all__ = ['fock', 'fock_dm', 'coherent', 'coherent_dm']


def fock(
    dim: int | tuple[int, ...],
    number: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        number _(int or tuple of ints)_: Fock state number of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock(3, 1)
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.fock((3, 2), (1, 0))
        tensor([[0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]])
    """
    # convert integer inputs to tuples by default, and check dimensions match
    dim = (dim,) if isinstance(dim, int) else dim
    number = (number,) if isinstance(number, int) else number
    if len(dim) != len(number):
        raise ValueError(
            'Arguments `number` must have the same length as `dim` of length'
            f' {len(dim)}, but has length {len(number)}.'
        )

    # compute the required basis state
    n = 0
    for d, s in zip(dim, number):
        n = d * n + s
    ket = torch.zeros(prod(dim), 1, dtype=get_cdtype(dtype), device=device)
    ket[n] = 1.0
    return ket


def fock_dm(
    dim: int | tuple[int, ...],
    number: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the density matrix of a Fock state or the density matrix of a tensor
    product of Fock states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        number _(int or tuple of ints)_: Fock state number of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock_dm(3, 1)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]])
        >>> dq.fock_dm((3, 2), (1, 0))
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    return todm(fock(dim, number, dtype=dtype, device=device))


def coherent(
    dim: int | tuple[int, ...],
    alpha: Number | ArrayLike,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a coherent state, or the ket of a tensor product of coherent
    states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alpha _(number or array-like object)_: Coherent state amplitude of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the coherent state.

    Examples:
        >>> dq.coherent(4, 0.5)
        tensor([[0.882+0.j],
                [0.441+0.j],
                [0.156+0.j],
                [0.047+0.j]])
        >>> dq.coherent((2, 3), (0.5, 0.5j))
        tensor([[ 0.775+0.000j],
                [ 0.000+0.386j],
                [-0.146+0.000j],
                [ 0.423+0.000j],
                [ 0.000+0.211j],
                [-0.080+0.000j]])
    """
    cdtype = get_cdtype(dtype)

    # convert inputs to tuples by default, and check dimensions match
    dim = (dim,) if isinstance(dim, int) else dim
    if isinstance(alpha, get_args(Number)):
        alpha = torch.as_tensor(alpha, dtype=cdtype, device=device)
    else:
        alpha = to_tensor(alpha, dtype=cdtype, device=device)

    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(-1)
    elif alpha.ndim > 1:
        raise ValueError(
            'Argument `alpha` must be a 0-D or 1-D array-like object, but is'
            f' a {alpha.ndim}-D object.'
        )
    if len(dim) != len(alpha):
        raise ValueError(
            'Arguments `alpha` must have the same length as `dim` of length'
            f' {len(dim)}, but has length {len(alpha)}.'
        )

    kets = [
        displace(d, a, dtype=cdtype, device=device)
        @ fock(d, 0, dtype=cdtype, device=device)
        for d, a in zip(dim, alpha)
    ]
    return tensprod(*kets)


def coherent_dm(
    dim: int | tuple[int, ...],
    alpha: Number | ArrayLike,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the density matrix of a coherent state, or the density matrix of a
    tensor product of coherent states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alpha _(number or array-like object)_: Coherent state amplitude of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the coherent state.

    Examples:
        >>> dq.coherent_dm(4, 0.5)
        tensor([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
                [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
                [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
                [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]])
        >>> dq.coherent_dm((2, 3), (0.5, 0.5))
        tensor([[0.600+0.j, 0.299+0.j, 0.113+0.j, 0.328+0.j, 0.163+0.j, 0.062+0.j],
                [0.299+0.j, 0.149+0.j, 0.056+0.j, 0.163+0.j, 0.081+0.j, 0.031+0.j],
                [0.113+0.j, 0.056+0.j, 0.021+0.j, 0.062+0.j, 0.031+0.j, 0.012+0.j],
                [0.328+0.j, 0.163+0.j, 0.062+0.j, 0.179+0.j, 0.089+0.j, 0.034+0.j],
                [0.163+0.j, 0.081+0.j, 0.031+0.j, 0.089+0.j, 0.044+0.j, 0.017+0.j],
                [0.062+0.j, 0.031+0.j, 0.012+0.j, 0.034+0.j, 0.017+0.j, 0.006+0.j]])
    """
    return todm(coherent(dim, alpha, dtype=dtype, device=device))
