from __future__ import annotations

from math import prod

import torch
from torch import Tensor

from .operators import displace
from .tensor_types import to_cdtype
from .utils import ket_to_dm

__all__ = ['fock', 'fock_dm', 'coherent', 'coherent_dm']


def fock(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dims: Dimension of the Hilbert space of each mode.
        states: Fock state of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock(3, 1)
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]], dtype=torch.complex128)
        >>> dq.fock((3, 2), (1, 0))
        tensor([[0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]], dtype=torch.complex128)
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
    ket = torch.zeros(prod(dims), 1, dtype=to_cdtype(dtype), device=device)
    ket[n] = 1.0
    return ket


def fock_dm(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Returns the density matrix of a Fock state or the density matrix of a tensor
    product of Fock states.

    Args:
        dims: Dimension of the Hilbert space of each mode.
        states: Fock state of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock_dm(3, 1)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]], dtype=torch.complex128)
        >>> dq.fock_dm((3, 2), (1, 0))
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                dtype=torch.complex128)
    """
    return ket_to_dm(fock(dims, states, dtype=to_cdtype(dtype), device=device))


def coherent(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Returns the ket of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Coherent state amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the coherent state.

    Examples:
        >>> dq.coherent(5, 0.2)
        tensor([[0.980+0.j],
                [0.196+0.j],
                [0.028+0.j],
                [0.003+0.j],
                [0.000+0.j]], dtype=torch.complex128)
    """
    cdtype = to_cdtype(dtype)
    return displace(dim, alpha, dtype=cdtype, device=device) @ fock(
        dim, 0, dtype=cdtype, device=device
    )


def coherent_dm(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Density matrix of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Coherent state amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the coherent state.

    Examples:
        >>> dq.coherent_dm(5, 0.2)
        tensor([[0.961+0.j, 0.192+0.j, 0.027+0.j, 0.003+0.j, 0.000+0.j],
                [0.192+0.j, 0.038+0.j, 0.005+0.j, 0.001+0.j, 0.000+0.j],
                [0.027+0.j, 0.005+0.j, 0.001+0.j, 0.000+0.j, 0.000+0.j],
                [0.003+0.j, 0.001+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]],
                dtype=torch.complex128)
    """
    return ket_to_dm(coherent(dim, alpha, dtype=to_cdtype(dtype), device=device))
