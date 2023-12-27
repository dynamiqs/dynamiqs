from __future__ import annotations

from cmath import exp as cexp
from math import prod, sqrt

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .tensor_types import get_cdtype
from .utils import dag, tensprod

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


def eye(*dims: int, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the identity operator.

    If only a single dimension is provided, `eye` returns the identity operator
    of corresponding dimension. If instead multiples dimensions are provided, `eye`
    returns the identity operator of the composite Hilbert space given by the product
    of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (n, n))_ Identity operator (with _n_ the product of
            dimensions in `dims`).

    Examples:
        >>> dq.eye(4)
        Array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]], dtype=complex64)
        >>> dq.eye(2, 3)
        Array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    dim = prod(dims)
    return jnp.eye(dim, dtype=dtype)


def zero(*dims: int, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the null operator.

    If only a single dimension is provided, `zero` returns the null operator
    of corresponding dimension. If instead multiples dimensions are provided, `zero`
    returns the null operator of the composite Hilbert space given by the product
    of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (n, n))_ Null operator (with _n_ the product of dimensions
            in `dims`).

    Examples:
        >>> dq.zero(4)
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        >>> dq.zero(2, 3)
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    dim = prod(dims)
    return jnp.zeros((dim, dim), dtype=dtype)


def destroy(
    *dims: int, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array | tuple[Array, ...]:
    r"""Returns a bosonic annihilation operator, or a tuple of annihilation operators in
    a multi-mode system.

    If only a single dimension is provided, `destroy` returns the annihilation operator
    of corresponding dimension. If instead multiples dimensions are provided, `destroy`
    returns a tuple of each annihilation operator of given dimension, in the Hilbert
    space given by the product of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned array(s).

    Returns:
        _(array or tuple of arrays)_ Annihilation operator of given dimension, or
            tuple of annihilation operators in a multi-mode system.

    Examples:
        >>> dq.destroy(4)
        Array([[0.   +0.j, 1.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 1.414+0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 1.732+0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j]], dtype=complex64)
        >>> a, b = dq.destroy(2, 3)
        >>> a
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        >>> b
        Array([[0.   +0.j, 1.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 1.414+0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 1.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 1.414+0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j]],      dtype=complex64)
    """  # noqa: E501
    dtype = get_cdtype(dtype)

    if len(dims) == 1:
        return _destroy_single(dims[0], dtype=dtype)

    a = [_destroy_single(dim, dtype=dtype) for dim in dims]
    I = [eye(dim, dtype=dtype) for dim in dims]
    return tuple(
        tensprod(*[a[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _destroy_single(dim: int, *, dtype: jnp.complex64 | jnp.complex128) -> Array:
    """Bosonic annihilation operator."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, stop=dim, dtype=dtype)), k=1)


def create(
    *dims: int, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array | tuple[Array, ...]:
    r"""Returns a bosonic creation operator, or a tuple of creation operators in a
    multi-mode system.

    If only a single dimension is provided, `create` returns the creation operator of
    corresponding dimension. If instead multiples dimensions are provided, `create`
    returns a tuple of each creation operator of given dimension, in the Hilbert space
    given by the product of all dimensions.

    Args:
        *dims: Variable length argument list of the Hilbert space dimensions.
        dtype: Complex data type of the returned array(s).

    Returns:
        _(array or tuple of arrays)_ Creation operator of given dimension, or tuple
            of creation operators in a multi-mode system.

    Examples:
        >>> dq.create(4)
        Array([[0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [1.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 1.414+0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 1.732+0.j, 0.   +0.j]], dtype=complex64)
        >>> a, b = dq.create(2, 3)
        >>> a
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        >>> b
        Array([[0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [1.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 1.414+0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 1.   +0.j, 0.   +0.j, 0.   +0.j],
               [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.   +0.j, 1.414+0.j, 0.   +0.j]],      dtype=complex64)
    """  # noqa: E501
    dtype = get_cdtype(dtype)

    if len(dims) == 1:
        return _create_single(dims[0], dtype=dtype)

    adag = [_create_single(dim, dtype=dtype) for dim in dims]
    I = [eye(dim, dtype=dtype) for dim in dims]
    return tuple(
        tensprod(*[adag[j] if i == j else I[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _create_single(dim: int, *, dtype: jnp.complex64 | jnp.complex128) -> Array:
    """Bosonic creation operator."""
    return jnp.diag(jnp.sqrt(jnp.arange(1, stop=dim, dtype=dtype)), k=-1)


def number(dim: int, *, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the number operator of a bosonic mode.

    It is defined by $n = a^\dag a$, where $a$ and $a^\dag$ are the annihilation and
    creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (dim, dim))_ Number operator.

    Examples:
        >>> dq.number(4)
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 3.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.diag(jnp.arange(0, stop=dim, dtype=dtype))


def parity(dim: int, *, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the parity operator of a bosonic mode.

    It is defined by $P = e^{i\pi a^\dag a}$, where $a$ and $a^\dag$ are the
    annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (dim, dim))_ Parity operator.

    Examples:
        >>> dq.parity(4)
        Array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    diag_values = jnp.ones(dim, dtype=dtype)
    diag_values = diag_values.at[1::2].set(-1)
    return jnp.diag(diag_values)


def displace(
    dim: int, alpha: ArrayLike, *, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array:
    r"""Returns the displacement operator of complex amplitude $\alpha$.

    It is defined by
    $$
        D(\alpha) = \exp(\alpha a^\dag - \alpha^* a),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(array_like of shape (...))_: Displacement amplitude.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (..., dim, dim))_ Displacement operator.

    Examples:
        >>> dq.displace(4, 0.5)
        Array([[ 0.882+0.j, -0.441+0.j,  0.156+0.j, -0.047+0.j],
               [ 0.441+0.j,  0.662+0.j, -0.542+0.j,  0.27 +0.j],
               [ 0.156+0.j,  0.542+0.j,  0.442+0.j, -0.697+0.j],
               [ 0.047+0.j,  0.27 +0.j,  0.697+0.j,  0.662+0.j]], dtype=complex64)
        >>> dq.displace(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    dtype = get_cdtype(dtype)

    alpha = jnp.asarray(alpha, dtype=dtype)
    alpha = alpha[..., None, None]  # (..., 1, 1)

    a = destroy(dim, dtype=dtype)  # (n, n)
    return jax.scipy.linalg.expm(alpha * dag(a) - alpha.conj() * a)


def squeeze(
    dim: int, z: ArrayLike, *, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array:
    r"""Returns the squeezing operator of complex squeezing amplitude $z$.

    It is defined by
    $$
        S(z) = \exp\left(\frac{1}{2}\left(z^* a^2 - z a^{\dag 2}\right)\right),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        z _(array_like of shape (...))_: Squeezing amplitude.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (..., dim, dim))_ Squeezing operator.

    Examples:
        >>> dq.squeeze(4, 0.5)
        Array([[ 0.938+0.j,  0.   +0.j,  0.346+0.j,  0.   +0.j],
               [ 0.   +0.j,  0.818+0.j,  0.   +0.j,  0.575+0.j],
               [-0.346+0.j,  0.   +0.j,  0.938+0.j,  0.   +0.j],
               [ 0.   +0.j, -0.575+0.j,  0.   +0.j,  0.818+0.j]], dtype=complex64)
        >>> dq.squeeze(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    dtype = get_cdtype(dtype)

    z = jnp.asarray(z, dtype=dtype)
    z = z[..., None, None]  # (..., 1, 1)

    a = destroy(dim, dtype=dtype)  # (n, n)
    a2 = a @ a
    return jax.scipy.linalg.expm(0.5 * (z.conj() * a2 - z * dag(a2)))


def quadrature(
    dim: int, phi: float, *, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array:
    r"""Returns the quadrature operator of phase angle $\phi$.

    It is defined by $x_\phi = (e^{i\phi} a^\dag + e^{-i\phi} a) / 2$, where $a$ and
    $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (dim, dim))_ Quadrature operator.

    Examples:
        >>> from math import pi
        >>> dq.quadrature(3, 0.0)
        Array([[0.   +0.j, 0.5  +0.j, 0.   +0.j],
               [0.5  +0.j, 0.   +0.j, 0.707+0.j],
               [0.   +0.j, 0.707+0.j, 0.   +0.j]], dtype=complex64)
        >>> dq.quadrature(3, pi / 2)
        Array([[0.+0.j   , 0.-0.5j  , 0.+0.j   ],
               [0.+0.5j  , 0.+0.j   , 0.-0.707j],
               [0.+0.j   , 0.+0.707j, 0.+0.j   ]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    a = destroy(dim, dtype=dtype)
    return 0.5 * (cexp(1.0j * phi) * dag(a) + cexp(-1.0j * phi) * a)


def position(dim: int, *, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the position operator $x = (a^\dag + a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (dim, dim))_ Position operator.

    Examples:
        >>> dq.position(3)
        Array([[0.   +0.j, 0.5  +0.j, 0.   +0.j],
               [0.5  +0.j, 0.   +0.j, 0.707+0.j],
               [0.   +0.j, 0.707+0.j, 0.   +0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    a = destroy(dim, dtype=dtype)
    return 0.5 * (a + dag(a))


def momentum(dim: int, *, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the momentum operator $p = i (a^\dag - a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (dim, dim))_ Momentum operator.

    Examples:
        >>> dq.momentum(3)
        Array([[0.+0.j   , 0.-0.5j  , 0.+0.j   ],
               [0.+0.5j  , 0.+0.j   , 0.-0.707j],
               [0.+0.j   , 0.+0.707j, 0.+0.j   ]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    a = destroy(dim, dtype=dtype)
    return 0.5j * (dag(a) - a)


def sigmax(*, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the Pauli $\sigma_x$ operator.

    It is defined by $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2, 2))_ Pauli $\sigma_x$ operator.

    Examples:
        >>> dq.sigmax()
        Array([[0.+0.j, 1.+0.j],
               [1.+0.j, 0.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)


def sigmay(*, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the Pauli $\sigma_y$ operator.

    It is defined by $\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2, 2))_ Pauli $\sigma_y$ operator.

    Examples:
        >>> dq.sigmay()
        Array([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)


def sigmaz(*, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the Pauli $\sigma_z$ operator.

    It is defined by $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2, 2))_ Pauli $\sigma_z$ operator.

    Examples:
        >>> dq.sigmaz()
        Array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)


def sigmap(*, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the Pauli raising operator $\sigma_+$.

    It is defined by $\sigma_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2, 2))_ Pauli $\sigma_+$ operator.

    Examples:
        >>> dq.sigmap()
        Array([[0.+0.j, 1.+0.j],
               [0.+0.j, 0.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)


def sigmam(*, dtype: jnp.complex64 | jnp.complex128 | None = None) -> Array:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    It is defined by $\sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$.

    Args:
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2, 2))_ Pauli $\sigma_-$ operator.

    Examples:
        >>> dq.sigmam()
        Array([[0.+0.j, 0.+0.j],
               [1.+0.j, 0.+0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    return jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)


def hadamard(
    n: int = 1, *, dtype: jnp.complex64 | jnp.complex128 | None = None
) -> Array:
    r"""Returns the Hadamard transform on $n$ qubits.

    For a single qubit, it is defined by
    $$
        H = \frac{1}{\sqrt2} \begin{pmatrix}
            1 & 1 \\\\
            1 & -1
        \end{pmatrix}
    $$
    For $n$ qubits, it is defined by the tensor product of Hadamard matrices:
    $$
        H_n = \bigotimes_{k=1}^n H
    $$

    Args:
        n: Number of qubits to act on.
        dtype: Complex data type of the returned array.

    Returns:
        _(array of shape (2^n, 2^n))_ Hadamard transform operator.

    Examples:
        >>> dq.hadamard()
        Array([[ 0.707+0.j,  0.707+0.j],
               [ 0.707+0.j, -0.707+0.j]], dtype=complex64)
        >>> dq.hadamard(2)
        Array([[ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j],
               [ 0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j],
               [ 0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j],
               [ 0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5-0.j]], dtype=complex64)
    """
    dtype = get_cdtype(dtype)
    H1 = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=dtype) / sqrt(2)
    Hs = jnp.broadcast_to(H1, (n, 2, 2))  # (n, 2, 2)
    return tensprod(*Hs)
