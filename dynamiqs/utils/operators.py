from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jax.typing import ArrayLike

from .._utils import cdtype
from ..qarrays import QArray, asqarray
from .utils import tensor

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


def eye(*dims: int) -> QArray:
    r"""Returns the identity operator.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns the
    identity operator of the composite Hilbert space of dimension $n=\prod n_k$:
    $$
        I_n = I_{n_1}\otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each subsystem.

    Returns:
        _(qarray of shape (n, n))_ Identity operator, with _n = prod(dims)_.

    Examples:
        Single-mode $I_4$:
        >>> dq.eye(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]

        Multi-mode $I_2 \otimes I_3$:
        >>> dq.eye(2, 3)
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """
    dim = prod(dims)
    array = jnp.eye(dim, dtype=cdtype())
    return asqarray(array, dims=dims)


def zero(*dims: int) -> QArray:
    r"""Returns the null operator.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns the
    null operator of the composite Hilbert space of dimension $n=\prod n_k$:
    $$
        0_n = 0_{n_1}\otimes\dots\otimes 0_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each subsystem.

    Returns:
        _(qarray of shape (n, n))_ Null operator, with _n = prod(dims)_.

    Examples:
        Single-mode $0_4$:
        >>> dq.zero(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

        Multi-mode $0_2 \otimes 0_3$:
        >>> dq.zero(2, 3)
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
    """
    dim = prod(dims)
    array = jnp.zeros((dim, dim), dtype=cdtype())
    return asqarray(array, dims=dims)


def destroy(*dims: int) -> QArray | tuple[QArray, ...]:
    r"""Returns a bosonic annihilation operator, or a tuple of annihilation operators
    for a multi-mode system.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns a
    tuple with _len(dims)_ operators $(A_1,\dots,A_N)$, where $A_k$ is the annihilation
    operator acting on the $k$-th subsystem within the composite Hilbert space of
    dimension $n=\prod n_k$:
    $$
        A_k = I_{n_1} \otimes\dots\otimes a_{n_k} \otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each mode.

    Returns:
        _(qarray or tuple of qarrays, each of shape (n, n))_ Annihilation operator(s),
            with _n = prod(dims)_.

    Examples:
        Single-mode $a$:
        >>> dq.destroy(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[0.   +0.j 1.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 1.414+0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 1.732+0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]]

        Mult-mode $a\otimes I_3$ and $I_2\otimes b$:
        >>> a, b = dq.destroy(2, 3)
        >>> a
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        >>> b
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[0.   +0.j 1.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 1.414+0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 1.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 1.414+0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]]
    """
    if len(dims) == 1:
        return _destroy_single(dims[0])

    a = [_destroy_single(dim) for dim in dims]
    Id = [eye(dim) for dim in dims]
    return tuple(
        tensor(*[a[j] if i == j else Id[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _destroy_single(dim: int) -> QArray:
    """Bosonic annihilation operator."""
    array = jnp.diag(jnp.sqrt(jnp.arange(1, stop=dim, dtype=cdtype())), k=1)
    return asqarray(array, dims=dim)


def create(*dims: int) -> QArray | tuple[QArray, ...]:
    r"""Returns a bosonic creation operator, or a tuple of creation operators for a
    multi-mode system.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns a
    tuple with _len(dims)_ operators $(A_1^\dag,\dots,A_N^\dag)$, where $A_k^\dag$ is
    the creation operator acting on the $k$-th subsystem within the composite Hilbert
    space of dimension $n=\prod n_k$:
    $$
        A_k^\dag = I_{n_1} \otimes\dots\otimes a_{n_k}^\dag \otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each mode.

    Returns:
        _(qarray or tuple of qarrays, each of shape (n, n))_ Creation operator(s), with
            _n = prod(dims)_.

    Examples:
        Single-mode $a^\dag$:
        >>> dq.create(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [1.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 1.414+0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 1.732+0.j 0.   +0.j]]

        Mult-mode $a^\dag\otimes I_3$ and $I_2\otimes b^\dag$:
        >>> adag, bdag = dq.create(2, 3)
        >>> adag
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        >>> bdag
        DenseQArray: shape=(6, 6), dims=(2, 3), dtype=complex64
        [[0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [1.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 1.414+0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 1.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 1.414+0.j 0.   +0.j]]
    """
    if len(dims) == 1:
        return _create_single(dims[0])

    adag = [_create_single(dim) for dim in dims]
    Id = [eye(dim) for dim in dims]
    return tuple(
        tensor(*[adag[j] if i == j else Id[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def _create_single(dim: int) -> QArray:
    """Bosonic creation operator."""
    array = jnp.diag(jnp.sqrt(jnp.arange(1, stop=dim, dtype=cdtype())), k=-1)
    return asqarray(array, dims=dim)


def number(dim: int | None = None) -> QArray:
    r"""Returns the number operator of a bosonic mode.

    It is defined by $n = a^\dag a$, where $a$ and $a^\dag$ are the annihilation and
    creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.

    Returns:
        _(qarray of shape (dim, dim))_ Number operator.

    Examples:
        >>> dq.number(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 2.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 3.+0.j]]
    """
    array = jnp.diag(jnp.arange(0, stop=dim, dtype=cdtype()))
    return asqarray(array)


def parity(dim: int) -> QArray:
    r"""Returns the parity operator of a bosonic mode.

    It is defined by $P = e^{i\pi a^\dag a}$, where $a$ and $a^\dag$ are the
    annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.

    Returns:
        _(qarray of shape (dim, dim))_ Parity operator.

    Examples:
        >>> dq.parity(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]]
    """
    diag_values = jnp.ones(dim, dtype=cdtype())
    diag_values = diag_values.at[1::2].set(-1)
    array = jnp.diag(diag_values)
    return asqarray(array)


def displace(dim: int, alpha: ArrayLike) -> QArray:
    r"""Returns the displacement operator of complex amplitude $\alpha$.

    It is defined by
    $$
        D(\alpha) = \exp(\alpha a^\dag - \alpha^* a),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(array_like of shape (...))_: Displacement amplitude.

    Returns:
        _(qarray of shape (..., dim, dim))_ Displacement operator.

    Examples:
        >>> dq.displace(4, 0.5)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[ 0.882+0.j -0.441+0.j  0.156+0.j -0.047+0.j]
         [ 0.441+0.j  0.662+0.j -0.542+0.j  0.27 +0.j]
         [ 0.156+0.j  0.542+0.j  0.442+0.j -0.697+0.j]
         [ 0.047+0.j  0.27 +0.j  0.697+0.j  0.662+0.j]]
        >>> dq.displace(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    alpha = jnp.asarray(alpha, dtype=cdtype())
    alpha = alpha[..., None, None]  # (..., 1, 1)
    a = destroy(dim)  # (n, n)
    return (alpha * a.dag() - alpha.conj() * a).expm()


def squeeze(dim: int, z: ArrayLike) -> QArray:
    r"""Returns the squeezing operator of complex squeezing amplitude $z$.

    It is defined by
    $$
        S(z) = \exp\left(\frac{1}{2}\left(z^* a^2 - z a^{\dag 2}\right)\right),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        z _(array_like of shape (...))_: Squeezing amplitude.

    Returns:
        _(qarray of shape (..., dim, dim))_ Squeezing operator.

    Examples:
        >>> dq.squeeze(4, 0.5)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[ 0.938+0.j  0.   +0.j  0.346+0.j  0.   +0.j]
         [ 0.   +0.j  0.818+0.j  0.   +0.j  0.575+0.j]
         [-0.346+0.j  0.   +0.j  0.938+0.j  0.   +0.j]
         [ 0.   +0.j -0.575+0.j  0.   +0.j  0.818+0.j]]
        >>> dq.squeeze(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    z = jnp.asarray(z, dtype=cdtype())
    z = z[..., None, None]  # (..., 1, 1)

    a = destroy(dim)  # (n, n)
    a2 = a @ a
    return (0.5 * (z.conj() * a2 - z * a2.dag())).expm()


def quadrature(dim: int, phi: float) -> QArray:
    r"""Returns the quadrature operator of phase angle $\phi$.

    It is defined by $x_\phi = (e^{i\phi} a^\dag + e^{-i\phi} a) / 2$, where $a$ and
    $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.

    Returns:
        _(qarray of shape (dim, dim))_ Quadrature operator.

    Examples:
        >>> dq.quadrature(3, 0.0)
        DenseQArray: shape=(3, 3), dims=(3,), dtype=complex64
        [[0.   +0.j 0.5  +0.j 0.   +0.j]
         [0.5  +0.j 0.   +0.j 0.707+0.j]
         [0.   +0.j 0.707+0.j 0.   +0.j]]
        >>> dq.quadrature(3, jnp.pi / 2)
        DenseQArray: shape=(3, 3), dims=(3,), dtype=complex64
        [[ 0.+0.j    -0.-0.5j    0.+0.j   ]
         [-0.+0.5j    0.+0.j    -0.-0.707j]
         [ 0.+0.j    -0.+0.707j  0.+0.j   ]]
    """
    a = destroy(dim)
    return 0.5 * (jnp.exp(1.0j * phi) * a.dag() + jnp.exp(-1.0j * phi) * a)


def position(dim: int) -> QArray:
    r"""Returns the position operator $x = (a^\dag + a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.

    Returns:
        _(qarray of shape (dim, dim))_ Position operator.

    Examples:
        >>> dq.position(3)
        DenseQArray: shape=(3, 3), dims=(3,), dtype=complex64
        [[0.   +0.j 0.5  +0.j 0.   +0.j]
         [0.5  +0.j 0.   +0.j 0.707+0.j]
         [0.   +0.j 0.707+0.j 0.   +0.j]]
    """
    a = destroy(dim)
    return 0.5 * (a + a.dag())


def momentum(dim: int) -> QArray:
    r"""Returns the momentum operator $p = i (a^\dag - a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.

    Returns:
        _(qarray of shape (dim, dim))_ Momentum operator.

    Examples:
        >>> dq.momentum(3)
        DenseQArray: shape=(3, 3), dims=(3,), dtype=complex64
        [[ 0.+0.j    -0.-0.5j    0.+0.j   ]
         [ 0.+0.5j    0.+0.j    -0.-0.707j]
         [ 0.+0.j     0.+0.707j  0.+0.j   ]]
    """
    a = destroy(dim)
    return 0.5j * (a.dag() - a)


def sigmax() -> QArray:
    r"""Returns the Pauli $\sigma_x$ operator.

    It is defined by $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_x$ operator.

    Examples:
        >>> dq.sigmax()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
    """
    array = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=cdtype())
    return asqarray(array)


def sigmay() -> QArray:
    r"""Returns the Pauli $\sigma_y$ operator.

    It is defined by $\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_y$ operator.

    Examples:
        >>> dq.sigmay()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
    """
    array = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=cdtype())
    return asqarray(array)


def sigmaz() -> QArray:
    r"""Returns the Pauli $\sigma_z$ operator.

    It is defined by $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_z$ operator.

    Examples:
        >>> dq.sigmaz()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[ 1.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
    """
    array = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=cdtype())
    return asqarray(array)


def sigmap() -> QArray:
    r"""Returns the Pauli raising operator $\sigma_+$.

    It is defined by $\sigma_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_+$ operator.

    Examples:
        >>> dq.sigmap()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[0.+0.j 1.+0.j]
         [0.+0.j 0.+0.j]]
    """
    array = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=cdtype())
    return asqarray(array)


def sigmam() -> QArray:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    It is defined by $\sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_-$ operator.

    Examples:
        >>> dq.sigmam()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[0.+0.j 0.+0.j]
         [1.+0.j 0.+0.j]]
    """
    array = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=cdtype())
    return asqarray(array)


def hadamard(n: int = 1) -> QArray:
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

    Returns:
        _(qarray of shape (2^n, 2^n))_ Hadamard transform operator.

    Examples:
        >>> dq.hadamard()
        DenseQArray: shape=(2, 2), dims=(2,), dtype=complex64
        [[ 0.707+0.j  0.707+0.j]
         [ 0.707+0.j -0.707+0.j]]
        >>> dq.hadamard(2)
        DenseQArray: shape=(4, 4), dims=(2, 2), dtype=complex64
        [[ 0.5+0.j  0.5+0.j  0.5+0.j  0.5+0.j]
         [ 0.5+0.j -0.5+0.j  0.5+0.j -0.5+0.j]
         [ 0.5+0.j  0.5+0.j -0.5+0.j -0.5+0.j]
         [ 0.5+0.j -0.5+0.j -0.5+0.j  0.5-0.j]]
    """
    H1 = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cdtype()) / jnp.sqrt(2)
    Hs = jnp.broadcast_to(H1, (n, 2, 2))  # (n, 2, 2)
    return tensor(*Hs)
