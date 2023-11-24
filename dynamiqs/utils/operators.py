from __future__ import annotations

from cmath import exp as cexp
from math import prod
from typing import get_args

import torch
from torch import Tensor

from .._utils import to_device
from .tensor_types import ArrayLike, Number, get_cdtype, to_tensor
from .utils import tensprod

__all__ = [
    'eye',
    'zero',
    'destroy',
    'create',
    'number',
    'parity',
    'displace',
    'squeeze',
    'quadrature',
    'position',
    'momentum',
    'sigmax',
    'sigmay',
    'sigmaz',
    'sigmap',
    'sigmam',
    'hadamard',
]


def eye(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
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
    return torch.eye(dim, dtype=get_cdtype(dtype), device=device)


def zero(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the null operator.

    If only a single dimension is provided, `zero` returns the null operator
    of corresponding dimension. If instead multiples dimensions are provided, `zero`
    returns the null operator of the composite Hilbert space given by the product
    of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Null operator (with _n_ the product of dimensions in `dims`).

    Examples:
        >>> dq.zero(4)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        >>> dq.zero(2, 3)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    dim = prod(dims)
    return torch.zeros(dim, dim, dtype=get_cdtype(dtype), device=device)


def destroy(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
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
    cdtype = get_cdtype(dtype)
    device = to_device(device)
    if len(dims) == 1:
        return _destroy_single(dims[0], dtype=cdtype, device=device)

    a = [_destroy_single(dim, dtype=cdtype, device=device) for dim in dims]
    I = [eye(dim, dtype=cdtype, device=device) for dim in dims]
    return tuple(
        tensprod(*[a[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _destroy_single(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128,
    device: torch.device,
) -> Tensor:
    """Bosonic annihilation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(1).to(dtype)


def create(
    *dims: int,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
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
    cdtype = get_cdtype(dtype)
    device = to_device(device)
    if len(dims) == 1:
        return _create_single(dims[0], dtype=cdtype, device=device)

    adag = [_create_single(dim, dtype=cdtype, device=device) for dim in dims]
    I = [eye(dim, dtype=cdtype, device=device) for dim in dims]
    return tuple(
        tensprod(*[adag[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _create_single(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128,
    device: torch.device,
) -> Tensor:
    """Bosonic creation operator."""
    return torch.arange(1, dim, device=device).sqrt().diag(-1).to(dtype)


def number(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the number operator of a bosonic mode.

    It is defined by $n = a^\dag a$, where $a$ and $a^\dag$ are the annihilation and
    creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Number operator.

    Examples:
        >>> dq.number(4)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 3.+0.j]])
    """
    return torch.arange(dim, device=device).diag().to(get_cdtype(dtype))


def parity(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the parity operator of a bosonic mode.

    It is defined by $P = e^{i\pi a^\dag a}$, where $a$ and $a^\dag$ are the
    annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Parity operator.

    Examples:
        >>> dq.parity(4)
        tensor([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
    """
    diag_values = torch.ones(dim, device=device, dtype=get_cdtype(dtype))
    diag_values[1::2] = -1
    return diag_values.diag()


def displace(
    dim: int,
    alpha: Number | ArrayLike,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the displacement operator of complex amplitude $\alpha$.

    It is defined by
    $$
        D(\alpha) = \exp(\alpha a^\dag - \alpha^* a),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(...)_: Displacement amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(..., dim, dim)_ Displacement operator.

    Examples:
        >>> dq.displace(4, 0.5)
        tensor([[ 0.882+0.j, -0.441+0.j,  0.156+0.j, -0.047+0.j],
                [ 0.441+0.j,  0.662+0.j, -0.542+0.j,  0.270+0.j],
                [ 0.156+0.j,  0.542+0.j,  0.442+0.j, -0.697+0.j],
                [ 0.047+0.j,  0.270+0.j,  0.697+0.j,  0.662+0.j]])
        >>> dq.displace(4, [0.1, 0.2]).shape
        torch.Size([2, 4, 4])
    """
    cdtype = get_cdtype(dtype)
    device = to_device(device)

    if isinstance(alpha, get_args(Number)):
        alpha = torch.as_tensor(alpha, dtype=cdtype, device=device)
    else:
        alpha = to_tensor(alpha, dtype=cdtype, device=device)
    alpha = alpha[..., None, None]  # (..., 1, 1)

    a = destroy(dim, dtype=cdtype, device=device)  # (n, n)
    return torch.matrix_exp(alpha * a.mH - alpha.conj() * a)


def squeeze(
    dim: int,
    z: Number | ArrayLike,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the squeezing operator of complex squeezing amplitude $z$.

    It is defined by
    $$
        S(z) = \exp\left(\frac{1}{2}\left(z^* a^2 - z a^{\dag 2}\right)\right),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        z _(...)_: Squeezing amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(..., dim, dim)_ Squeezing operator.

    Examples:
        >>> dq.squeeze(4, 0.5)
        tensor([[ 0.938+0.j,  0.000+0.j,  0.346+0.j,  0.000+0.j],
                [ 0.000+0.j,  0.818+0.j,  0.000+0.j,  0.575+0.j],
                [-0.346+0.j,  0.000+0.j,  0.938+0.j,  0.000+0.j],
                [ 0.000+0.j, -0.575+0.j,  0.000+0.j,  0.818+0.j]])
        >>> dq.squeeze(4, [0.1, 0.2]).shape
        torch.Size([2, 4, 4])
    """
    cdtype = get_cdtype(dtype)
    device = to_device(device)

    if isinstance(z, get_args(Number)):
        z = torch.as_tensor(z, dtype=cdtype, device=device)
    else:
        z = to_tensor(z, dtype=cdtype, device=device)
    z = z[..., None, None]  # (..., 1, 1)

    a = destroy(dim, dtype=dtype, device=device)  # (n, n)
    a2 = a @ a
    return torch.matrix_exp(0.5 * (z.conj() * a2 - z * a2.mH))


def quadrature(
    dim: int,
    phi: float,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the quadrature operator of phase angle $\phi$.

    It is defined by $x_\phi = (e^{i\phi} a^\dag + e^{-i\phi} a) / 2$, where $a$ and
    $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Quadrature operator.

    Examples:
        >>> from math import pi
        >>> dq.quadrature(3, 0.0)
        tensor([[0.000+0.j, 0.500+0.j, 0.000+0.j],
                [0.500+0.j, 0.000+0.j, 0.707+0.j],
                [0.000+0.j, 0.707+0.j, 0.000+0.j]])
        >>> dq.quadrature(3, pi / 2)
        tensor([[    0.000+0.000j,     0.000-0.500j,     0.000+0.000j],
                [    0.000+0.500j,     0.000+0.000j,     0.000-0.707j],
                [    0.000+0.000j,     0.000+0.707j,     0.000+0.000j]])
    """
    a = destroy(dim, dtype=dtype, device=device)
    return 0.5 * (cexp(1.0j * phi) * a.mH + cexp(-1.0j * phi) * a)


def position(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the position operator $x = (a^\dag + a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Position operator.

    Examples:
        >>> dq.position(3)
        tensor([[0.000+0.j, 0.500+0.j, 0.000+0.j],
                [0.500+0.j, 0.000+0.j, 0.707+0.j],
                [0.000+0.j, 0.707+0.j, 0.000+0.j]])
    """
    a = destroy(dim, dtype=dtype, device=device)
    return 0.5 * (a + a.mH)


def momentum(
    dim: int,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the momentum operator $p = i (a^\dag - a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(dim, dim)_ Momentum operator.

    Examples:
        >>> dq.momentum(3)
        tensor([[0.+0.000j, -0.-0.500j, 0.+0.000j],
                [0.+0.500j, 0.+0.000j, -0.-0.707j],
                [0.+0.000j, 0.+0.707j, 0.+0.000j]])
    """
    a = destroy(dim, dtype=dtype, device=device)
    return 0.5j * (a.mH - a)


def sigmax(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_x$ operator.

    It is defined by $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $\sigma_x$ operator.

    Examples:
        >>> dq.sigmax()
        tensor([[0.+0.j, 1.+0.j],
                [1.+0.j, 0.+0.j]])
    """
    return torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=get_cdtype(dtype), device=device
    )


def sigmay(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_y$ operator.

    It is defined by $\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $\sigma_y$ operator.

    Examples:
        >>> dq.sigmay()
        tensor([[0.+0.j, -0.-1.j],
                [0.+1.j, 0.+0.j]])
    """
    return torch.tensor(
        [[0.0, -1.0j], [1.0j, 0.0]], dtype=get_cdtype(dtype), device=device
    )


def sigmaz(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli $\sigma_z$ operator.

    It is defined by $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2, 2)_ Pauli $\sigma_z$ operator.

    Examples:
        >>> dq.sigmaz()
        tensor([[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]])
    """
    return torch.tensor(
        [[1.0, 0.0], [0.0, -1.0]], dtype=get_cdtype(dtype), device=device
    )


def sigmap(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli raising operator $\sigma_+$.

    It is defined by $\sigma_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

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
    return torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=get_cdtype(dtype), device=device
    )


def sigmam(
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    It is defined by $\sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$.

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
    return torch.tensor(
        [[0.0, 0.0], [1.0, 0.0]], dtype=get_cdtype(dtype), device=device
    )


def hadamard(
    n: int = 1,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the Hadamard transform on `n` qubits

    It is defined by $H_0=1$ and
    $$
        H_m = \frac{1}{\sqrt2} \begin{pmatrix}
            H_{m-1} & H_{m-1} \\
            H_{m-1} & -H_{m-1}
        \end{pmatrix}
    $$

    Args:
        n: integer number of qubits to act on
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(2^n, 2^n)_ Hadamard transform operator.

    Examples:
        >>> dq.hadamard()
        tensor([[ 0.707,  0.707],
                [ 0.707, -0.707]])

    """
    if n == 0:
        return torch.Tensor([1.0], dtype=dtype, device=device)

    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device) * (2**-0.5)
    if n == 1:
        return H1

    return tensprod(hadamard(n - 1, dtype=dtype, device=device), H1)
