from math import prod
from typing import Tuple, Union

import torch
from operators import displace
from torch import Tensor, device, dtype
from utils import ket_to_dm

__all__ = ['fock', 'fock_dm', 'coherent', 'coherent_dm']


def fock(
    dims: Union[int, Tuple[int, ...]],
    states: Union[int, Tuple[int, ...]],
    *,
    dtype: dtype = torch.complex128,
    device: device = None,
) -> Tensor:
    """Generate the state vector of a single-mode Fock state, or of a tensor product of
    Fock states.

    Example:
        >>> tq.fock(3, 1)
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]], dtype=torch.complex128)
        >>> tq.fock((3, 2), (1, 0))
        tensor([[0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]], dtype=torch.complex128)
    """
    # Convert integer inputs to tuples by default, and check dimensions match
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(states, int):
        states = (states,)
    if len(dims) != len(states):
        raise ValueError(
            f'Arguments `dims` ({len(dims)}) and `states` ({len(states)}) do not have'
            ' the same number of elements.'
        )

    # compute the required basis state
    n = 0
    for dim, state in zip(dims, states):
        n = dim * n + state
    ket = torch.zeros(prod(dims), 1, dtype=dtype, device=device)
    ket[n] = 1.0
    return ket


def fock_dm(
    dims: Union[int, Tuple[int, ...]],
    states: Union[int, Tuple[int, ...]],
    *,
    dtype: dtype = torch.complex128,
    device: device = None,
) -> Tensor:
    """Generate the density matrix of a single-mode Fock state, or of a tensor product
    of Fock states.

    Example:
        >>> tq.fock_dm(3, 1)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]], dtype=torch.complex128)
        >>> tq.fock_dm((3, 2), (1, 0))
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                dtype=torch.complex128)
    """
    return ket_to_dm(fock(dims, states, dtype=dtype, device=device))


def coherent(
    dim: int, alpha: complex, *, dtype: dtype = torch.complex128, device: device = None
) -> Tensor:
    """Generate the state vector of a single-mode coherent state.

    Example:
        >>> tq.coherent(5, 0.2)
        tensor([[0.980+0.j],
                [0.196+0.j],
                [0.028+0.j],
                [0.003+0.j],
                [0.000+0.j]], dtype=torch.complex128)
    """
    vac = fock(dim, 0)
    D = displace(dim, alpha, dtype=dtype, device=device)
    return D @ vac


def coherent_dm(
    dim: int, alpha: complex, *, dtype: dtype = torch.complex128, device: device = None
) -> Tensor:
    """Generate the density matrix of a single-mode coherent state.

    Example:
        >>> tq.coherent(5, 0.2)
        tensor([[0.961+0.j, 0.192+0.j, 0.027+0.j, 0.003+0.j, 0.000+0.j],
                [0.192+0.j, 0.038+0.j, 0.005+0.j, 0.001+0.j, 0.000+0.j],
                [0.027+0.j, 0.005+0.j, 0.001+0.j, 0.000+0.j, 0.000+0.j],
                [0.003+0.j, 0.001+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]],
                dtype=torch.complex128)
    """
    return ket_to_dm(coherent(dim, alpha, dtype=dtype, device=device))
