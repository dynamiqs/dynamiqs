from __future__ import annotations

from functools import reduce

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

__all__ = [
    'dag',
    'mpow',
    'trace',
    'ptrace',
    'tensprod',
    'expect',
    'norm',
    'unit',
    'dissipator',
    'lindbladian',
    'isket',
    'isbra',
    'isdm',
    'isop',
    'toket',
    'tobra',
    'todm',
    'braket',
    'overlap',
    'fidelity',
]


def dag(x: ArrayLike) -> Array:
    r"""Returns the adjoint (complex conjugate transpose) of a ket, bra, density matrix
    or operator.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.

    Returns:
       _(array of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Adjoint of `x`.

    Notes:
        This function is equivalent to `x.mT.conj()`.

    Examples:
        >>> dq.fock(2, 0)
        Array([[1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.dag(dq.fock(2, 0))
        Array([[1.-0.j, 0.-0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    return x.mT.conj()


def mpow(x: ArrayLike, n: int) -> Array:
    """Returns the $n$-th matrix power of an array.

    Notes:
        This function is equivalent to `jnp.linalg.matrix_power(x, n)`.

    Args:
        x _(array_like of shape (..., n, n))_: Square matrix.
        n: Integer exponent.

    Returns:
        _(array of shape (..., n, n))_ Matrix power of `x`.

    Examples:
        >>> dq.mpow(dq.sigmax(), 2)
        Array([[1.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    return jnp.linalg.matrix_power(x, n)


def trace(x: ArrayLike) -> Array:
    r"""Returns the trace of an array along its last two dimensions.

    Args:
        x _(array_like of shape (..., n, n))_: Array.

    Returns:
        _(array of shape (...))_ Trace of `x`.

    Examples:
        >>> x = jnp.ones((3, 3))
        >>> dq.trace(x)
        Array(3., dtype=float32)
    """
    x = jnp.asarray(x)
    return x.trace(axis1=-1, axis2=-2)


def _hdim(x: ArrayLike) -> int:
    x = jnp.asarray(x)
    return x.shape[-2] if isket(x) else x.shape[-1]


def ptrace(x: ArrayLike, keep: int | tuple[int, ...], dims: tuple[int, ...]) -> Array:
    r"""Returns the partial trace of a ket, bra or density matrix.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra or
            density matrix of a composite system.
        keep _(int or tuple of ints)_: Dimensions to keep after partial trace.
        dims _(tuple of ints)_: Dimensions of each subsystem in the composite system
            Hilbert space tensor product.

    Returns:
        _(array of shape (..., m, m))_ Density matrix (with `m <= n`).

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.
        ValueError: If `dims` does not match the shape of `x`, or if `keep` is
            incompatible with `dims`.

    Notes:
        The returned object is always a density matrix, even if the input is a ket or a
        bra.

    Examples:
        >>> psi_abc = dq.tensprod(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi_abc.shape
        (60, 1)
        >>> rho_a = dq.ptrace(psi_abc, 0, (3, 4, 5))
        >>> rho_a.shape
        (3, 3)
        >>> rho_bc = dq.ptrace(psi_abc, (1, 2), (3, 4, 5))
        >>> rho_bc.shape
        (20, 20)
    """
    x = jnp.asarray(x)

    # convert keep and dims to arrays
    keep = jnp.asarray([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
    dims = jnp.asarray(dims)  # e.g. [20, 2, 5]
    ndims = len(dims)  # e.g. 3

    # check that input dimensions match
    hdim = _hdim(x)
    prod_dims = jnp.prod(dims)
    if not prod_dims == hdim:
        dims_prod_str = '*'.join(str(d.item()) for d in dims) + f'={prod_dims}'
        raise ValueError(
            'Argument `dims` must match the Hilbert space dimension of `x` of'
            f' {hdim}, but the product of its values is {dims_prod_str}.'
        )
    if jnp.any(keep < 0) or jnp.any(keep > len(dims) - 1):
        raise ValueError(
            'Argument `keep` must match the Hilbert space structure specified by'
            ' `dims`.'
        )

    # sort keep
    keep = keep.sort()

    # create einsum alphabet
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # compute einsum equations
    eq1 = alphabet[:ndims]  # e.g. 'abc'
    unused = iter(alphabet[ndims:])
    eq2 = [next(unused) if i in keep else eq1[i] for i in range(ndims)]  # e.g. 'ade'

    # trace out x over unkept dimensions
    bshape = x.shape[:-2]
    if isket(x) or isbra(x):
        x = x.reshape(*bshape, *dims)  # e.g. (..., 20, 2, 5)
        eq = ''.join(['...'] + eq1 + [',...'] + eq2)  # e.g. '...abc,...ade'
        x = jnp.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    elif isdm(x):
        x = x.reshape(*bshape, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = ''.join(['...'] + eq1 + eq2)  # e.g. '...abcade'
        x = jnp.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {x.shape}.'
        )

    # reshape to final dimension
    nkeep = jnp.prod(dims[keep])  # e.g. 10
    return x.reshape(*bshape, nkeep, nkeep)  # e.g. (..., 10, 10)


def tensprod(*args: ArrayLike) -> Array:
    r"""Returns the tensor product of multiple kets, bras, density matrices or
    operators.

    The returned array shape is:

    - $(..., n, 1)$ with $n=\prod_k n_k$ if all input arrays are kets with shape
      $(..., n_k, 1)$,
    - $(..., 1, n)$ with $n=\prod_k n_k$ if all input arrays are bras with shape
      $(..., 1, n_k)$,
    - $(..., n, n)$ with $n=\prod_k n_k$ if all input arrays are density matrices or
      operators with shape $(..., n_k, n_k)$.

    Notes:
        This function is the equivalent of `qutip.tensor()`.

    Args:
        *args _(array_like of shape (..., n_k, 1) or (..., 1, n_k) or (..., n_k, n_k))_:
            Variable length argument list of kets, bras, density matrices or operators.

    Returns:
        _(array of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Tensor product of
            the input arrays.

    Examples:
        >>> psi = dq.tensprod(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi.shape
        (60, 1)
    """
    args = [jnp.asarray(arg) for arg in args]

    # todo: use jax.lax.reduce
    return reduce(_bkron, args)


# batched Kronecker product of two arrays
_bkron = jnp.vectorize(jnp.kron, signature='(a,b),(c,d)->(ac,bd)')


def expect(O: ArrayLike, x: ArrayLike) -> Array:
    r"""Returns the expectation value of an operator on a ket, bra or density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a ket $\ket\psi$ or bra $\bra\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned array is complex-valued. If the operator $O$ corresponds to a
        physical observable, it is Hermitian: $O^\dag=O$, and the expectation value
        is real. One can then keep only the real values of the returned array using
        `dq.expect(O, x).real`.

    Args:
        O _(array_like of shape (n, n))_: Arbitrary operator.
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra or
            density matrix.

    Returns:
        _(array of shape (...))_ Complex-valued expectation value.

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.

    Examples:
        >>> a = dq.destroy(16)
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.expect(dq.dag(a) @ a, psi)
        Array(4.+0.j, dtype=complex64)
    """
    O = jnp.asarray(O)
    x = jnp.asarray(x)

    if isket(x):
        return (dag(x) @ O @ x).squeeze((-1, -2))  # <x|O|x>
    elif isbra(x):
        return (x @ O @ dag(x)).squeeze((-1, -2))
    elif isdm(x):
        return trace(O @ x)  # tr(Ox)
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {x.shape}.'
        )


def norm(x: ArrayLike) -> Array:
    r"""Returns the norm of a ket, bra or density matrix.

    For a ket or a bra, the returned norm is $\sqrt{\braket{\psi|\psi}}$. For a density
    matrix, it is $\tr{\rho}$.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra or
            density matrix.

    Returns:
        _(array of shape (...))_ Real-valued norm of `x`.

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.

    Examples:
        For a ket:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(psi)
        Array(1.414, dtype=float32)

        For a density matrix:
        >>> rho = dq.fock_dm(4, 0) + dq.fock_dm(4, 1) + dq.fock_dm(4, 2)
        >>> dq.norm(rho)
        Array(3., dtype=float32)
    """
    x = jnp.asarray(x)

    if isket(x) or isbra(x):
        return jnp.linalg.norm(x, axis=(-1, -2)).real
    elif isdm(x):
        return trace(x).real
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {x.shape}.'
        )


def unit(x: ArrayLike) -> Array:
    r"""Normalize a ket, bra or density matrix to unit norm.

    The returned object is divided by its norm (see [`dq.norm()`][dynamiqs.norm]).

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra or
            density matrix.

    Returns:
        _(array of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Normalized ket,
            bra or density matrix.

    Examples:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(psi)
        Array(1.414, dtype=float32)
        >>> psi = dq.unit(psi)
        >>> dq.norm(psi)
        Array(1., dtype=float32)
    """
    x = jnp.asarray(x)

    return x / norm(x)[..., None, None]


def dissipator(L: ArrayLike, rho: ArrayLike) -> Array:
    r"""Applies the Lindblad dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:
    $$
        \mathcal{D}[L] (\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L _(array_like of shape (..., n, n))_: Jump operator.
        rho _(array_like of shape (..., n, n))_: Density matrix.

    Returns:
        _(array of shape (..., n, n))_ Resulting operator (it is not a density matrix).

    Examples:
        >>> L = dq.destroy(4)
        >>> rho = dq.fock_dm(4, 2)
        >>> dq.dissipator(L, rho)
        Array([[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  2.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j, -2.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]], dtype=complex64)
    """
    L = jnp.asarray(L)
    rho = jnp.asarray(rho)

    Ldag = dag(L)
    LdagL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * LdagL @ rho - 0.5 * rho @ LdagL


def lindbladian(H: ArrayLike, jump_ops: ArrayLike, rho: ArrayLike) -> Array:
    r"""Applies the Lindbladian superoperator to a density matrix.

    The Lindbladian superoperator $\mathcal{L}$ is defined by:
    $$
        \mathcal{L} (\rho) = -i[H,\rho] + \sum_{k=1}^N \mathcal{D}[L_k] (\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of $N$ jump operators
    (arbitrary operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator
    (see [`dq.dissipator()`][dynamiqs.dissipator]).

    Notes:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H _(array_like of shape (..., n, n))_: Hamiltonian.
        jump_ops _(array_like of shape (N, ..., n, n))_: Sequence of jump operators.
        rho _(array_like of shape (..., n, n))_: Density matrix.

    Returns:
        _(array of shape (..., n, n))_ Resulting operator (it is not a density matrix).

    Examples:
        >>> a = dq.destroy(4)
        >>> H = dq.dag(a) @ a
        >>> L = [a, dq.dag(a) @ a]
        >>> rho = dq.fock_dm(4, 1)
        >>> dq.lindbladian(H, L, rho)
        Array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]], dtype=complex64)
    """  # noqa: E501
    H = jnp.asarray(H)
    jump_ops = jnp.asarray(jump_ops)
    rho = jnp.asarray(rho)

    return -1j * (H @ rho - rho @ H) + dissipator(jump_ops, rho).sum(0)


def isket(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of a ket.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.isket(jnp.ones((3, 1)))
        True
        >>> dq.isket(jnp.ones((5, 3, 1)))
        True
        >>> dq.isket(jnp.ones((3, 3)))
        False
    """
    x = jnp.asarray(x)
    return x.shape[-1] == 1


def isbra(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of a bra.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the second to last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.isbra(jnp.ones((1, 3)))
        True
        >>> dq.isbra(jnp.ones((5, 1, 3)))
        True
        >>> dq.isbra(jnp.ones((3, 3)))
        False
    """
    x = jnp.asarray(x)
    return x.shape[-2] == 1


def isdm(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of a density matrix.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.

    Examples:
        >>> dq.isdm(jnp.ones((3, 3)))
        True
        >>> dq.isdm(jnp.ones((5, 3, 3)))
        True
        >>> dq.isdm(jnp.ones((3, 1)))
        False
    """
    x = jnp.asarray(x)
    return x.shape[-1] == x.shape[-2]


def isop(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of an operator.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.

    Examples:
        >>> dq.isop(jnp.ones((3, 3)))
        True
        >>> dq.isop(jnp.ones((5, 3, 3)))
        True
        >>> dq.isop(jnp.ones((3, 1)))
        False
    """
    x = jnp.asarray(x)
    return x.shape[-1] == x.shape[-2]


def toket(x: ArrayLike) -> Array:
    r"""Returns the ket representation of a pure quantum state.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(array of shape (..., n, 1))_ Ket.

    Examples:
        >>> psi = dq.tobra(dq.fock(3, 0))  # shape: (1, 3)
        >>> psi
        Array([[1.-0.j, 0.-0.j, 0.-0.j]], dtype=complex64)
        >>> dq.toket(psi)  # shape: (3, 1)
        Array([[1.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)

    if isbra(x):
        return dag(x)
    elif isket(x):
        return x
    else:
        raise ValueError(f'Argument `x` must be a ket or bra, but has shape {x.shape}.')


def tobra(x: ArrayLike) -> Array:
    r"""Returns the bra representation of a pure quantum state.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(array of shape (..., 1, n))_ Bra.

    Examples:
        >>> psi = dq.fock(3, 0)  # shape: (3, 1)
        >>> psi
        Array([[1.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.tobra(psi)  # shape: (1, 3)
        Array([[1.-0.j, 0.-0.j, 0.-0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)

    if isbra(x):
        return x
    elif isket(x):
        return dag(x)
    else:
        raise ValueError(f'Argument `x` must be a ket or bra, but has shape {x.shape}.')


def todm(x: ArrayLike) -> Array:
    r"""Returns the density matrix representation of a quantum state.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra or
            density matrix.

    Returns:
        _(array of shape (..., n, n))_ Density matrix.

    Examples:
        >>> psi = dq.fock(3, 0)  # shape: (3, 1)
        >>> psi
        Array([[1.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.todm(psi)  # shape: (3, 3)
        Array([[1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)

    if isbra(x):
        return dag(x) @ x
    elif isket(x):
        return x @ dag(x)
    elif isdm(x):
        return x
    else:
        raise ValueError(
            'Argument `x` must be a ket, bra or density matrix, but has shape'
            f' {x.shape}.'
        )


def braket(x: ArrayLike, y: ArrayLike) -> Array:
    r"""Returns the inner product $\braket{\psi|\varphi}$ between two kets.

    Args:
        x (array_like of shape _(..., n, 1))_: Left-side ket.
        y (array_like of shape _(..., n, 1))_: Right-side ket.

    Returns:
        _(array of shape (...))_ Complex-valued inner product.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> fock01 = dq.unit(dq.fock(3, 0) + dq.fock(3, 1))
        >>> dq.braket(fock0, fock01)
        Array(0.707+0.j, dtype=complex64)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if not isket(x):
        raise ValueError(f'Argument `x` must be a ket but has shape {x.shape}.')
    if not isket(y):
        raise ValueError(f'Argument `y` must be a ket but has shape {y.shape}.')

    return (dag(x) @ y).squeeze((-1, -2))


def overlap(x: ArrayLike, y: ArrayLike) -> Array:
    r"""Returns the overlap between two quantum states.

    The overlap is computed

    - as $\lvert\braket{\psi|\varphi}\rvert^2$ if both arguments are kets $\ket\psi$
      and $\ket\varphi$,
    - as $\lvert\bra\psi \rho \ket\psi\rvert$ if one argument is a ket $\ket\psi$ and
      the other is a density matrix $\rho$,
    - as $\tr{\rho^\dag\sigma}$ if both arguments are density matrices $\rho$ and
      $\sigma$.

    Args:
        x _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        y _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued overlap.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.overlap(fock0, fock0)
        Array(1., dtype=float32)
        >>> fock01_dm = 0.5 * (dq.fock_dm(3, 0) + dq.fock_dm(3, 1))
        >>> dq.overlap(fock0, fock01_dm)
        Array(0.5, dtype=float32)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if not isket(x) and not isdm(x):
        raise ValueError(
            f'Argument `x` must be a ket or density matrix, but has shape {x.shape}.'
        )
    if not isket(y) and not isdm(y):
        raise ValueError(
            f'Argument `y` must be a ket or density matrix, but has shape {y.shape}.'
        )

    if isket(x) and isket(y):
        return jnp.abs((dag(x) @ y).squeeze((-1, -2))) ** 2
    elif isket(x):
        return jnp.abs((dag(x) @ y @ x).squeeze((-1, -2)))
    elif isket(y):
        return jnp.abs((dag(y) @ x @ y).squeeze((-1, -2)))
    else:
        return trace(dag(x) @ y).squeeze((-1, -2)).real


def fidelity(x: ArrayLike, y: ArrayLike) -> Array:
    r"""Returns the fidelity of two states, kets or density matrices.

    The fidelity is computed

    - as $F(\ket\psi,\ket\varphi)=\left|\braket{\psi|\varphi}\right|^2$ if both
      arguments are kets,
    - as $F(\ket\psi,\rho)=\lvert\braket{\psi|\rho|\psi}\rvert$ if one arguments is a
      ket and the other is a density matrix,
    - as $F(\rho,\sigma)=\tr{\sqrt{\sqrt\rho\sigma\sqrt\rho}}^2$ if both arguments are
      density matrices.

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Args:
        x _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        y _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued fidelity.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.fidelity(fock0, fock0)
        Array(1., dtype=float32)
        >>> fock01_dm = 0.5 * (dq.fock_dm(3, 1) + dq.fock_dm(3, 0))
        >>> dq.fidelity(fock01_dm, fock01_dm)
        Array(1., dtype=float32)
        >>> dq.fidelity(fock0, fock01_dm)
        Array(0.5, dtype=float32)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if isket(x) or isket(y):
        return overlap(x, y)
    else:
        return _dm_fidelity(x, y)


def _dm_fidelity(x: ArrayLike, y: ArrayLike) -> Array:
    # returns the fidelity of two density matrices: Tr[sqrt(sqrt(x) @ y @ sqrt(x))]^2
    # x: (..., n, n), y: (..., n, n) -> (...)
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    sqrtm_x = _bsqrtm(x)
    tmp = sqrtm_x @ y @ sqrtm_x

    # we don't need the whole matrix `sqrtm(tmp)`, just its trace, which can be computed
    # by summing the square roots of `tmp` eigenvalues
    eigvals_tmp = jnp.linalg.eigvalsh(tmp)
    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    eigvals_tmp = jnp.where(eigvals_tmp < 0, 0, eigvals_tmp)
    trace_sqrtm_tmp = jnp.sqrt(eigvals_tmp).sum(-1)

    return (trace_sqrtm_tmp**2).real


# batched matrix square root
_bsqrtm = jnp.vectorize(jax.scipy.linalg.sqrtm, signature='(m,n)->(m,n)')
