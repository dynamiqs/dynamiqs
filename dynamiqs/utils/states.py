from __future__ import annotations

from math import prod

import torch
from torch import Tensor

from .operators import displace
from .tensor_types import ArrayLike, get_cdtype, to_tensor
from .utils import tensprod, todm

__all__ = ['fock', 'fock_dm', 'coherent', 'coherent_dm']


def fock(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        states _(int or tuple of ints)_: Fock state of each mode.
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
    dims = (dims,) if isinstance(dims, int) else dims
    states = (states,) if isinstance(states, int) else states
    if len(dims) != len(states):
        raise ValueError(
            'Arguments `states` must have the same length as `dims` of length'
            f' {len(dims)}, but has length {len(states)}.'
        )

    # compute the required basis state
    n = 0
    for dim, state in zip(dims, states):
        n = dim * n + state
    ket = torch.zeros(prod(dims), 1, dtype=get_cdtype(dtype), device=device)
    ket[n] = 1.0
    return ket


def fock_dm(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the density matrix of a Fock state or the density matrix of a tensor
    product of Fock states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        states _(int or tuple of ints)_: Fock state of each mode.
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
    return todm(fock(dims, states, dtype=dtype, device=device))


def coherent(
    dims: int | tuple[int, ...],
    alphas: int | float | complex | tuple[int | float | complex, ...] | ArrayLike,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a coherent state, or the ket of a tensor product of coherent
    states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alphas _(number or array-like object)_: Coherent state amplitude of each mode.
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
    dims = (dims,) if isinstance(dims, int) else dims
    if isinstance(alphas, (int, float, complex, tuple)):
        alphas = torch.as_tensor(alphas, dtype=cdtype, device=device)
    else:
        alphas = to_tensor(alphas, dtype=cdtype, device=device)

    if alphas.ndim == 0:
        alphas = alphas.unsqueeze(-1)
    elif alphas.ndim > 1:
        raise ValueError(
            'Argument `alphas` must be a 0-D or 1-D array-like object, but is'
            f'a {alphas.ndim}-D object.'
        )
    if len(dims) != len(alphas):
        raise ValueError(
            'Arguments `alphas` must have the same length as `dims` of length'
            f' {len(dims)}, but has length {len(alphas)}.'
        )

    kets = [
        displace(dim, alpha, dtype=cdtype, device=device)
        @ fock(dim, 0, dtype=cdtype, device=device)
        for dim, alpha in zip(dims, alphas)
    ]
    return tensprod(*kets)


def coherent_dm(
    dims: int | tuple[int, ...],
    alphas: complex | tuple[complex, ...] | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the density matrix of a coherent state, or the density matrix of a
    tensor product of coherent states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alphas _(complex, tuple of complex, or Tensor)_: Coherent state amplitude of
            each mode.
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
        >>> dq.coherent_dm((2, 3), (0.5, 0.5j))
        tensor([[ 0.600+0.000j,  0.000-0.299j, -0.113+0.000j,  0.328+0.000j,
                  0.000-0.163j, -0.062+0.000j],
                [ 0.000+0.299j,  0.149+0.000j, -0.000-0.056j,  0.000+0.163j,
                  0.081+0.000j, -0.000-0.031j],
                [-0.113+0.000j, -0.000+0.056j,  0.021+0.000j, -0.062+0.000j,
                 -0.000+0.031j,  0.012+0.000j],
                [ 0.328+0.000j,  0.000-0.163j, -0.062+0.000j,  0.179+0.000j,
                  0.000-0.089j, -0.034+0.000j],
                [ 0.000+0.163j,  0.081+0.000j, -0.000-0.031j,  0.000+0.089j,
                  0.044+0.000j, -0.000-0.017j],
                [-0.062+0.000j, -0.000+0.031j,  0.012+0.000j, -0.034+0.000j,
                 -0.000+0.017j,  0.006+0.000j]])
    """
    return todm(coherent(dims, alphas, dtype=dtype, device=device))
