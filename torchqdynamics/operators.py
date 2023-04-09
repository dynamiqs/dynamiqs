from __future__ import annotations

from math import prod

import torch
from torch import Tensor, device, dtype

from .utils import _extract_tuple_from_varargs, tensprod

__all__ = [
    'sigmax',
    'sigmay',
    'sigmaz',
    'sigmap',
    'sigmam',
    'qeye',
    'destroy',
    'create',
    'displace',
    'squeeze',
]


def sigmax(*, dtype: dtype = torch.complex128, device: device | None = None) -> Tensor:
    """Pauli X operator."""
    return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)


def sigmay(*, dtype: dtype = torch.complex128, device: device | None = None) -> Tensor:
    """Pauli Y operator."""
    return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)


def sigmaz(*, dtype: dtype = torch.complex128, device: device | None = None) -> Tensor:
    """Pauli Z operator."""
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)


def sigmap(*, dtype: dtype = torch.complex128, device: device | None = None) -> Tensor:
    """Pauli raising operator."""
    return torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)


def sigmam(*, dtype: dtype = torch.complex128, device: device | None = None) -> Tensor:
    """Pauli lowering operator."""
    return torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)


def qeye(
    *dims: int, dtype: dtype = torch.complex128, device: device | None = None
) -> Tensor:
    """Identity operator."""
    dims = _extract_tuple_from_varargs(dims)
    dim = prod(dims)
    return torch.eye(dim, dtype=dtype, device=device)


def destroy(
    *dims: int, dtype: dtype = torch.complex128, device: device = None
) -> Tensor | tuple[Tensor, ...]:
    """Bosonic annihilation operator.

    If only a single dimension is provided, `destroy` returns the annihilation operator
    of corresponding dimension. If instead multiples dimensions are provided, `destroy`
    returns a tuple of each annihilation operator of given dimension, in the Hilbert
    space given by the product of all dimensions.

    Example:
        >>> tq.destroy(4)
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.732+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]],
                dtype=torch.complex128)
        >>> a, b = tq.destroy(2, 3)
        >>> a
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                dtype=torch.complex128)
        >>> b
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.414+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]],
                dtype=torch.complex128)
    """
    dims = _extract_tuple_from_varargs(dims)

    # compute first destroy operator
    a = _destroy_single(dims[0], dtype=dtype, device=device)
    if len(dims) == 1:
        return a

    # compute all annihilation operators
    ops = [a]
    eye = qeye(dims[0], dtype=dtype, device=device)
    for i, dim in enumerate(dims[1:]):
        # single mode operators
        _a = _destroy_single(dim, dtype=dtype, device=device)
        _eye = qeye(dim, dtype=dtype, device=device)

        # update ops
        ops.append(tensprod(eye, _a))
        for j in range(i + 1):
            ops[j] = tensprod(ops[j], _eye)

        # update eye
        eye = tensprod(eye, _eye)

    return tuple(ops)


def _destroy_single(
    dim: int, *, dtype: dtype = torch.complex128, device: device | None = None
) -> Tensor:
    """Bosonic annihilation operator of a single mode."""
    return torch.diag(torch.sqrt(torch.arange(1, dim, device=device)), 1).to(dtype)


def create(
    *dims: int, dtype: dtype = torch.complex128, device: device | None = None
) -> Tensor | tuple[Tensor, ...]:
    """Bosonic creation operator.

    If only a single dimension is provided, `create` returns the creation operator of
    corresponding dimension. If instead multiples dimensions are provided, `create`
    returns a tuple of each creation operator of given dimension, in the Hilbert space
    given by the product of all dimensions.

    Example:
        >>> tq.create(4)
        tensor([[0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.732+0.j, 0.000+0.j]],
                dtype=torch.complex128)
        >>> a, b = tq.create(2, 3)
        >>> a
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                dtype=torch.complex128)
        >>> b
        tensor([[0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j]],
                dtype=torch.complex128)
    """
    dims = _extract_tuple_from_varargs(dims)

    # compute first destroy operator
    adag = _create_single(dims[0], dtype=dtype, device=device)
    if len(dims) == 1:
        return adag

    # compute all creation operators
    ops = [adag]
    eye = qeye(dims[0], dtype=dtype, device=device)
    for i, dim in enumerate(dims[1:]):
        # single mode operators
        _adag = _create_single(dim, dtype=dtype, device=device)
        _eye = qeye(dim, dtype=dtype, device=device)

        # update ops
        ops.append(tensprod(eye, _adag))
        for j in range(i + 1):
            ops[j] = tensprod(ops[j], _eye)

        # update eye
        eye = tensprod(eye, _eye)

    return tuple(ops)


def _create_single(
    dim: int, *, dtype: dtype = torch.complex128, device: device | None = None
) -> Tensor:
    """Bosonic creation operator of a single mode."""
    return torch.diag(torch.sqrt(torch.arange(1, dim, device=device)), -1).to(dtype)


def displace(
    dim: int,
    alpha: complex,
    *,
    dtype: dtype = torch.complex128,
    device: device | None = None,
) -> Tensor:
    """Single-mode displacement operator."""
    a = destroy(dim, dtype=dtype, device=device)
    return torch.matrix_exp(alpha * a.adjoint() - alpha.conjugate() * a)


def squeeze(
    dim: int,
    z: complex,
    *,
    dtype: dtype = torch.complex128,
    device: device | None = None,
) -> Tensor:
    """Single-mode displacement operator."""
    a = destroy(dim, dtype=dtype, device=device)
    a2 = a @ a
    return torch.matrix_exp(0.5 * (z.conjugate() * a2 - z * a2.adjoint()))
