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
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_x$ operator.

    It is defined by:

    $$
        \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}.
    $$

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $X$ operator.

    Examples:
        >>> dq.sigmax()
        tensor([[0.+0.j, 1.+0.j],
                [1.+0.j, 0.+0.j]])
    """
    return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmay(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_y$ operator.

    It is defined by:

    $$
        \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}.
    $$

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $Y$ operator.

    Examples:
        >>> dq.sigmay()
        tensor([[0.+0.j, -0.-1.j],
                [0.+1.j, 0.+0.j]])
    """
    return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmaz(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_z$ operator.

    It is defined by:

    $$
        \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.
    $$

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $Z$ operator.

    Examples:
        >>> dq.sigmaz()
        tensor([[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]])
    """
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)


@complex_tensor
def sigmap(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli raising operator $\sigma_+$.

    It is defined by:

    $$
        \sigma_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}.
    $$

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $\sigma_+$ operator.

    Examples:
        >>> dq.sigmap()
        tensor([[0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j]])
    """
    return torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def sigmam(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    It is defined by:

    $$
        \sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.
    $$

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $\sigma_-$ operator.

    Examples:
        >>> dq.sigmam()
        tensor([[0.+0.j, 0.+0.j],
                [1.+0.j, 0.+0.j]])
    """
    return torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)


@complex_tensor
def eye(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the identity operator.

    If only a single dimension is provided, `eye` returns the identity operator
    of corresponding dimension. If instead multiples dimensions are provided, `eye`
    returns the identity operator of the composite Hilbert space given by the product
    of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Identity operator (with _n_ the product of dimensions in `dims`).

    Examples:
        >>> dq.eye(4)
        tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
        >>> dq.eye(2, 3)
        tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
    """
    dim = prod(dims)
    return torch.eye(dim, dtype=dtype, device=device)


@complex_tensor
def destroy(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Returns a bosonic annihilation operator, or a tuple of annihilation operators in
    a multi-mode system.

    If only a single dimension is provided, `destroy` returns the annihilation operator
    of corresponding dimension. If instead multiples dimensions are provided, `destroy`
    returns a tuple of each annihilation operator of given dimension, in the Hilbert
    space given by the product of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(tensor or tuple of tensors)_ Annihilation operator of given dimension, or
            tuple of annihilation operators in a multi-mode system.

    Examples:
        >>> dq.destroy(4)
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.732+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]])
        >>> a, b = dq.destroy(2, 3)
        >>> a
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        >>> b
        tensor([[0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.414+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j]])
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
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Bosonic annihilation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(1).to(dtype)


@complex_tensor
def create(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Returns a bosonic creation operator, or a tuple of creation operators in a
    multi-mode system.

    If only a single dimension is provided, `create` returns the creation operator of
    corresponding dimension. If instead multiples dimensions are provided, `create`
    returns a tuple of each creation operator of given dimension, in the Hilbert space
    given by the product of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(tensor or tuple of tensors)_ Creation operator of given dimension, or tuple
            of creation operators in a multi-mode system.

    Examples:
        >>> dq.create(4)
        tensor([[0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 1.732+0.j, 0.000+0.j]])
        >>> a, b = dq.create(2, 3)
        >>> a
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        >>> b
        tensor([[0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 1.414+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j],
                [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.414+0.j, 0.000+0.j]])
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
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Bosonic creation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(-1).to(dtype)


@complex_tensor
def displace(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the displacement operator of complex amplitude $\alpha$.

    It is defined by

    $$
        D(\alpha) = \exp(\alpha a^\dag - \alpha^* a),
    $$

    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Displacement amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Displacement operator.

    Examples:
        >>> dq.displace(4, 0.5)
        tensor([[ 0.882+0.j, -0.441+0.j,  0.156+0.j, -0.047+0.j],
                [ 0.441+0.j,  0.662+0.j, -0.542+0.j,  0.270+0.j],
                [ 0.156+0.j,  0.542+0.j,  0.442+0.j, -0.697+0.j],
                [ 0.047+0.j,  0.270+0.j,  0.697+0.j,  0.662+0.j]])
    """
    a = destroy(dim, dtype=dtype, device=device)
    alpha = torch.as_tensor(alpha)
    return torch.matrix_exp(alpha * a.mH - alpha.conj() * a)


@complex_tensor
def squeeze(
    dim: int,
    z: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Returns the squeezing operator of complex squeezing amplitude $z$.

    It is defined by

    $$
        S(z) = \exp\left(\frac{1}{2}\left(z^* a^2 - z a^{\dag 2}\right)\right),
    $$

    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        z: Squeezing amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Squeezing operator.

    Examples:
        >>> dq.squeeze(4, 0.5)
        tensor([[ 0.938+0.j,  0.000+0.j,  0.346+0.j,  0.000+0.j],
                [ 0.000+0.j,  0.818+0.j,  0.000+0.j,  0.575+0.j],
                [-0.346+0.j,  0.000+0.j,  0.938+0.j,  0.000+0.j],
                [ 0.000+0.j, -0.575+0.j,  0.000+0.j,  0.818+0.j]])
    """
    a = destroy(dim, dtype=dtype, device=device)
    a2 = a @ a
    z = torch.as_tensor(z)
    return torch.matrix_exp(0.5 * (z.conj() * a2 - z * a2.mH))
