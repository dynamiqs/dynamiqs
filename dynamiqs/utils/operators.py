from __future__ import annotations

from math import prod

import torch
from torch import Tensor

from .tensor_types import complex_tensor
from .utils import tensprod

__all__ = [
    'sigmax',
    'sigmay',
    'sigmaz',
    'sigmap',
    'sigmam',
    'eye',
    'destroy',
    'create',
    'displace',
    'squeeze',
]


@complex_tensor
def sigmax(
    *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Returns the Pauli $X$ operator.

    Args:
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmay(
    *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Returns the Pauli $Y$ operator.

    Args:
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmaz(
    *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Returns the Pauli $Z$ operator.

    Args:
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)


@complex_tensor
def sigmap(
    *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    r"""Returns the Pauli raising operator $\sigma_+$.

    Args:
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    return torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmam(
    *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    Args:
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    return torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def eye(
    *dims: int, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Returns the identity operator.

    If only a single dimension is provided, `eye` returns the identity operator
    of corresponding dimension. If instead multiples dimensions are provided, `eye`
    returns the identity operator of the composite Hilbert space given by the product
    of all dimensions.

    Args:
        dims: Dimension of the Hilbert space.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
    """
    dim = prod(dims)
    return torch.eye(dim, dtype=dtype, device=device)


@complex_tensor
def destroy(
    *dims: int, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor | tuple[Tensor, ...]:
    """Returns a bosonic annihilation operator, or a tuple of annihilation operators in
    a multi-mode system.

    If only a single dimension is provided, `destroy` returns the annihilation operator
    of corresponding dimension. If instead multiples dimensions are provided, `destroy`
    returns a tuple of each annihilation operator of given dimension, in the Hilbert
    space given by the product of all dimensions.

    Args:
        dims: Dimension of the Hilbert space.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        Annihilation operator of given dimension, or tuple of annihilation operators in
            a multi-mode system.

    Examples:
        >>> dq.destroy(4)
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.732+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]],
                dtype=torch.complex128)
        >>> a, b = dq.destroy(2, 3)
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
    if len(dims) == 1:
        return _destroy_single(dims[0], dtype=dtype, device=device)

    a = [_destroy_single(dim, dtype=dtype, device=device) for dim in dims]
    I = [eye(dim, dtype=dtype, device=device) for dim in dims]
    return tuple(
        tensprod(*[a[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


@complex_tensor
def _destroy_single(
    dim: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Bosonic annihilation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(1).to(dtype)


@complex_tensor
def create(
    *dims: int, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor | tuple[Tensor, ...]:
    """Returns a bosonic creation operator, or a tuple of creation operators in a
    multi-mode system.

    If only a single dimension is provided, `create` returns the creation operator of
    corresponding dimension. If instead multiples dimensions are provided, `create`
    returns a tuple of each creation operator of given dimension, in the Hilbert space
    given by the product of all dimensions.

    Args:
        dims: Dimension of the Hilbert space.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        Creation operator of given dimension, or tuple of creation operators in a
            multi-mode system.

    Examples:
        >>> dq.create(4)
        tensor([[0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.732+0.j, 0.000+0.j]],
                dtype=torch.complex128)
        >>> a, b = dq.create(2, 3)
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
    if len(dims) == 1:
        return _create_single(dims[0], dtype=dtype, device=device)

    adag = [_create_single(dim, dtype=dtype, device=device) for dim in dims]
    I = [eye(dim, dtype=dtype, device=device) for dim in dims]
    return tuple(
        tensprod(*[adag[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


@complex_tensor
def _create_single(
    dim: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Bosonic creation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(-1).to(dtype)


@complex_tensor
def displace(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the displacement operator of amplitude $\alpha$.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Displacement amplitude.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        Displacement operator.
    """
    a = destroy(dim, dtype=dtype, device=device)
    alpha = torch.as_tensor(alpha)
    return torch.matrix_exp(alpha * a.mH - alpha.conj() * a)


@complex_tensor
def squeeze(
    dim: int,
    z: complex | Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Returns the squeezing operator of squeezing amplitude $z$.

    Args:
        dim: Dimension of the Hilbert space.
        z: Squeezing amplitude.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        Squeezing operator.
    """
    a = destroy(dim, dtype=dtype, device=device)
    a2 = a @ a
    z = torch.as_tensor(z)
    return torch.matrix_exp(0.5 * (z.conj() * a2 - z * a2.mH))
