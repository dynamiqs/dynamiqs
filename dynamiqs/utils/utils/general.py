from __future__ import annotations

from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from ..._checks import check_shape
from ..._utils import on_cpu

__all__ = [
    'dag',
    'powm',
    'expm',
    'cosm',
    'sinm',
    'trace',
    'tracemm',
    'ptrace',
    'tensor',
    'expect',
    'norm',
    'unit',
    'dissipator',
    'lindbladian',
    'isket',
    'isbra',
    'isdm',
    'isop',
    'isherm',
    'toket',
    'tobra',
    'todm',
    'proj',
    'braket',
    'overlap',
    'fidelity',
    'entropy_vn',
]


def dag(x: ArrayLike) -> Array:
    r"""Returns the adjoint (complex conjugate transpose) of a matrix.

    Args:
        x _(array_like of shape (..., m, n))_: Matrix.

    Returns:
       _(array of shape (..., n, m))_ Adjoint of `x`.

    Note-: Equivalent JAX syntax
        This function is equivalent to `x.mT.conj()`.

    Examples:
        >>> dq.fock(2, 0)
        Array([[1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.dag(dq.fock(2, 0))
        Array([[1.-0.j, 0.-0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., m, n)')
    return x.mT.conj()


def powm(x: ArrayLike, n: int) -> Array:
    """Returns the $n$-th matrix power of an array.

    Args:
        x _(array_like of shape (..., n, n))_: Square matrix.
        n: Integer exponent.

    Returns:
        _(array of shape (..., n, n))_ Matrix power of `x`.

    Note-: Equivalent JAX syntax
        This function is equivalent to `jnp.linalg.matrix_power(x, n)`.

    Examples:
        >>> dq.powm(dq.sigmax(), 2)
        Array([[1.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return jnp.linalg.matrix_power(x, n)


def expm(x: ArrayLike, *, max_squarings: int = 16) -> Array:
    """Returns the matrix exponential of an array.

    The exponential is computed using the scaling-and-squaring approximation method.

    Args:
        x _(array_like of shape (..., n, n))_: Square matrix.
        max_squarings: Number of squarings.

    Returns:
        _(array of shape (..., n, n))_ Matrix exponential of `x`.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        `jnp.scipy.linalg.expm(x, max_squarings=max_squarings)`.

    Examples:
        >>> dq.expm(dq.sigmaz())
        Array([[2.718+0.j, 0.   +0.j],
               [0.   +0.j, 0.368+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return jax.scipy.linalg.expm(x, max_squarings=max_squarings)


def cosm(x: ArrayLike) -> Array:
    r"""Returns the cosine of an array.

    Args:
        x _(array_like of shape (..., n, n))_: Square matrix.

    Returns:
        _(array of shape (..., n, n))_ Cosine of `x`.

    Note:
        This function uses [`jax.scipy.linalg.expm()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html)
        to compute the cosine of a matrix $A$:
        $$
            \cos(A) = \frac{e^{iA} + e^{-iA}}{2}
        $$

    Examples:
        >>> dq.cosm(jnp.pi * dq.sigmax())
        Array([[-1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return 0.5 * (expm(1j * x) + expm(-1j * x))


def sinm(x: ArrayLike) -> Array:
    r"""Returns the sine of an array.

    Args:
        x _(array_like of shape (..., n, n))_: Square matrix.

    Returns:
        _(array of shape (..., n, n))_ Sine of `x`.

    Note:
        This function uses [`jax.scipy.linalg.expm()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html)
        to compute the sine of a matrix $A$:
        $$
            \sin(A) = \frac{e^{iA} - e^{-iA}}{2i}
        $$

    Examples:
        >>> dq.sinm(0.5 * jnp.pi * dq.sigmax())
        Array([[0.-0.j, 1.-0.j],
               [1.-0.j, 0.-0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return -0.5j * (expm(1j * x) - expm(-1j * x))


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
    check_shape(x, 'x', '(..., n, n)')
    return x.trace(axis1=-1, axis2=-2)


def tracemm(x: ArrayLike, y: ArrayLike) -> Array:
    r"""Return the trace of a matrix multiplication using a fast implementation.

    The trace is computed as `sum(x * y.T)` where `*` is the element-wise product,
    instead of `trace(x @ y)` where `@` is the matrix product. Indeed, we have:

    $$
        \tr{xy} = \sum_i (xy)_{ii}
                = \sum_{i,j} x_{ij} y_{ji}
                = \sum_{i,j} x_{ij} (y^\intercal)_{ij}
                = \sum_{i,j} (x * y^\intercal)_{ij}
    $$

    Note:
        The resulting time complexity for $n\times n$ matrices is $\mathcal{O}(n^2)$
        instead of $\mathcal{O}(n^3)$ with the naive formula.

    Args:
        x _(array_like of shape (..., n, n))_: Array.
        y _(array_like of shape (..., n, n))_: Array.

    Returns:
        _(array of shape (...))_ Trace of `x @ y`.

    Examples:
        >>> x = jnp.ones((3, 3))
        >>> y = jnp.ones((3, 3))
        >>> dq.tracemm(x, y)
        Array(9., dtype=float32)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    check_shape(x, 'x', '(..., n, n)')
    check_shape(y, 'y', '(..., n, n)')
    return (x * y.mT).sum((-2, -1))


def _hdim(x: ArrayLike) -> int:
    x = jnp.asarray(x)
    return x.shape[-2] if isket(x) else x.shape[-1]


# @partial(jax.jit, static_argnums=(1, 2))
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

    Note:
        The returned object is always a density matrix, even if the input is a ket or a
        bra.

    Examples:
        >>> psi_abc = dq.tensor(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    # convert keep and dims to numpy arrays
    keep = np.asarray([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
    dims = np.asarray(dims)  # e.g. [20, 2, 5]
    ndims = len(dims)  # e.g. 3

    # check that input dimensions match
    hdim = _hdim(x)
    prod_dims = np.prod(dims)
    if prod_dims != hdim:
        dims_prod_str = '*'.join(str(d) for d in dims) + f'={prod_dims}'
        raise ValueError(
            'Argument `dims` must match the Hilbert space dimension of `x` of'
            f' {hdim}, but the product of its values is {dims_prod_str}.'
        )
    if np.any(keep < 0) or np.any(keep > len(dims) - 1):
        raise ValueError(
            'Argument `keep` must match the Hilbert space structure specified by'
            ' `dims`.'
        )

    # sort keep
    keep.sort()

    # create einsum alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # compute einsum equations
    eq1 = alphabet[:ndims]  # e.g. 'abc'
    unused = iter(alphabet[ndims:])
    eq2 = ''.join(
        [next(unused) if i in keep else eq1[i] for i in range(ndims)]
    )  # e.g. 'ade'

    bshape = x.shape[:-2]

    # trace out x over unkept dimensions
    if isket(x) or isbra(x):
        x = x.reshape(*bshape, *dims)  # e.g. (..., 20, 2, 5)
        eq = f'...{eq1},...{eq2}'  # e.g. '...abc,...ade'
        x = jnp.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    else:
        x = x.reshape(*bshape, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = f'...{eq1}{eq2}'  # e.g. '...abcade'
        x = jnp.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)

    nkeep = np.prod(dims[keep])  # e.g. 10
    return x.reshape(*bshape, nkeep, nkeep)  # e.g. (..., 10, 10)


def tensor(*args: ArrayLike) -> Array:
    r"""Returns the tensor product of multiple kets, bras, density matrices or
    operators.

    The returned array shape is:

    - $(..., n, 1)$ with $n=\prod_k n_k$ if all input arrays are kets with shape
      $(..., n_k, 1)$,
    - $(..., 1, n)$ with $n=\prod_k n_k$ if all input arrays are bras with shape
      $(..., 1, n_k)$,
    - $(..., n, n)$ with $n=\prod_k n_k$ if all input arrays are density matrices or
      operators with shape $(..., n_k, n_k)$.

    Args:
        *args _(array_like of shape (..., n_k, 1) or (..., 1, n_k) or (..., n_k, n_k))_:
            Variable length argument list of kets, bras, density matrices or operators.

    Returns:
        _(array of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Tensor product of
            the input arrays.

    Examples:
        >>> psi = dq.tensor(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi.shape
        (60, 1)
    """
    args = [jnp.asarray(arg) for arg in args]

    # TODO: use jax.lax.reduce
    return reduce(_bkron, args)


# batched Kronecker product of two arrays
_bkron = jnp.vectorize(jnp.kron, signature='(a,b),(c,d)->(ac,bd)')


def expect(O: ArrayLike, x: ArrayLike) -> Array:
    r"""Returns the expectation value of an operator or list of operators on a ket, bra
    or density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a ket $\ket\psi$ or bra $\bra\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned array is complex-valued. If the operator $O$ corresponds to a
        physical observable, it is Hermitian: $O^\dag=O$, and the expectation value
        is real. One can then keep only the real values of the returned array using
        `dq.expect(O, x).real`.

    Args:
        O _(array_like of shape (nO?, n, n))_: Arbitrary operator or list of _nO_
            operators.
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket,
            bra or density matrix.

    Returns:
        _(array of shape (nO?, ...))_ Complex-valued expectation value.

    Raises:
        ValueError: If `x` is not a ket, bra or density matrix.

    Examples:
        >>> O = dq.number(16)
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.expect(O, psi)
        Array(4.+0.j, dtype=complex64)
        >>> psis = [dq.fock(16, i) for i in range(5)]
        >>> dq.expect(O, psis).shape
        (5,)
        >>> Os = [dq.position(16), dq.momentum(16)]
        >>> dq.expect(Os, psis).shape
        (2, 5)
    """
    O = jnp.asarray(O)
    x = jnp.asarray(x)
    check_shape(O, 'O', '(?, n, n)', subs={'?': 'nO?'})
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    f = _expect_single
    if O.ndim > 2:
        f = jax.vmap(f, in_axes=(0, None))
    return f(O, x)


def _expect_single(O: Array, x: Array) -> Array:
    # O: (n, n), x: (..., n, m)
    if isket(x):
        return (dag(x) @ O @ x).squeeze((-1, -2))  # <x|O|x>
    elif isbra(x):
        return (x @ O @ dag(x)).squeeze((-1, -2))
    else:
        return tracemm(O, x)  # tr(Ox)


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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if isket(x) or isbra(x):
        return jnp.linalg.norm(x, axis=(-1, -2)).real
    else:
        return trace(x).real


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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

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
    check_shape(L, 'L', '(..., n, n)')
    check_shape(rho, 'rho', '(..., n, n)')

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

    Note:
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
    """
    H = jnp.asarray(H)
    jump_ops = jnp.asarray(jump_ops)
    rho = jnp.asarray(rho)
    check_shape(H, 'H', '(..., n, n)')
    check_shape(jump_ops, 'jump_ops', '(N, ..., n, n)')
    check_shape(rho, 'rho', '(..., n, n)')

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


def isherm(x: ArrayLike, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    r"""Returns True if the array is Hermitian.

    Args:
        x _(array_like of shape (..., n, n))_: Array.
        rtol: Relative tolerance of the check.
        atol: Absolute tolerance of the check.

    Returns:
        True if `x` is Hermitian, False otherwise.

    Examples:
        >>> dq.isherm(jnp.eye(3))
        Array(True, dtype=bool)
        >>> dq.isherm([[0, 1j], [1j, 0]])
        Array(False, dtype=bool)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return jnp.allclose(x, dag(x), rtol=rtol, atol=atol)


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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return dag(x)
    else:
        return x


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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return x
    else:
        return dag(x)


def todm(x: ArrayLike) -> Array:
    r"""Returns the density matrix representation of a quantum state.

    Note:
        This function is an alias of [`dq.proj()`][dynamiqs.proj]. If `x` is already a
        density matrix, it is returned directly.

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
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if isbra(x) or isket(x):
        return proj(x)
    else:
        return x


def proj(x: ArrayLike) -> Array:
    r"""Returns the projection operator onto a pure quantum state.

    The projection operator onto the state $\ket\psi$ is defined as
    $P_{\ket\psi} = \ket\psi\bra\psi$.

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(array of shape (..., n, n))_ Projection operator.

    Examples:
        >>> psi = dq.fock(3, 0)
        >>> dq.proj(psi)
        Array([[1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return dag(x) @ x
    else:
        return x @ dag(x)


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
    check_shape(x, 'x', '(..., n, 1)')
    check_shape(y, 'y', '(..., n, 1)')

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
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')
    check_shape(y, 'y', '(..., n, 1)', '(..., n, n)')

    if isket(x) and isket(y):
        return jnp.abs((dag(x) @ y).squeeze((-1, -2))) ** 2
    elif isket(x):
        return jnp.abs((dag(x) @ y @ x).squeeze((-1, -2)))
    elif isket(y):
        return jnp.abs((dag(y) @ x @ y).squeeze((-1, -2)))
    else:
        return tracemm(dag(x), y).real


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
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')
    check_shape(y, 'y', '(..., n, 1)', '(..., n, n)')

    if isket(x) or isket(y):
        return overlap(x, y)
    elif on_cpu(x):
        return _dm_fidelity_cpu(x, y)
    else:
        return _dm_fidelity_gpu(x, y)


def _dm_fidelity_cpu(x: Array, y: Array) -> Array:
    # returns the fidelity of two density matrices: Tr[sqrt(sqrt(x) @ y @ sqrt(x))]^2
    # x: (..., n, n), y: (..., n, n) -> (...)

    # Note that the fidelity can be rewritten as Tr[sqrt(x @ y)]^2 (see
    # https://arxiv.org/abs/2211.02623). This is cheaper numerically, we just compute
    # the eigenvalues w_i of the matrix product x @ y and compute the fidelity as
    # F = (\sum_i \sqrt{w_i})^2.

    # This method only works on CPU because we use `jnp.linalg.eigvals` which is only
    # implemented on the CPU backend (see https://github.com/google/jax/issues/1259
    # tracking support on GPU). Note that we can't use eigvalsh here, because the
    # matrix x @ y is not Hermitian.

    # note that we can't use `eigvalsh` here because x @ y is not necessarily Hermitian
    w = jnp.linalg.eigvals(x @ y).real
    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    w = jnp.where(w < 0, 0, w)
    return jnp.sqrt(w).sum(-1) ** 2


def _dm_fidelity_gpu(x: Array, y: Array) -> Array:
    # returns the fidelity of two density matrices: Tr[sqrt(sqrt(x) @ y @ sqrt(x))]^2
    # x: (..., n, n), y: (..., n, n) -> (...)

    # We compute the eigenvalues w_i of the matrix product sqrt(x) @ y @ sqrt(x)
    # and compute the fidelity as F = (\sum_i \sqrt{w_i})^2.

    sqrtm_x = _sqrtm_gpu(x)
    w = jnp.linalg.eigvalsh(sqrtm_x @ y @ sqrtm_x)
    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    w = jnp.where(w < 0, 0, w)
    return jnp.sqrt(w).sum(-1) ** 2


def _sqrtm_gpu(x: Array) -> Array:
    # returns the square root of a symmetric or Hermitian positive definite matrix
    # x: (..., n, n) -> (..., n, n)

    # code inspired by
    # https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228

    # This method is a GPU-compatible implementation of `jax.scipy.linalg.sqrtm` for
    # symmetric or Hermitian positive definite matrix.

    w, v = jnp.linalg.eigh(x)
    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    w = jnp.where(w < 0, 0, w)
    # numerical trick to compute 'v @ jnp.diag(jnp.sqrt(w)) @ v.mT.conj()' faster with
    # broadcasting
    return (v * jnp.sqrt(w)[None, :]) @ v.mT.conj()


def entropy_vn(x: ArrayLike) -> Array:
    r"""Returns the Von Neumann entropy of a ket or density matrix.

    It is defined by $S(\rho) = -\tr{\rho \ln \rho}$.

    Args:
        x _(array_like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued Von Neumann entropy.

    Examples:
        >>> rho = dq.unit(dq.fock_dm(2, 0) + dq.fock_dm(2, 1))
        >>> dq.entropy_vn(rho)
        Array(0.693, dtype=float32)
        >>> psis = [dq.fock(16, i) for i in range(5)]
        >>> dq.entropy_vn(psis).shape
        (5,)
    """
    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')

    if isket(x):
        return jnp.zeros(x.shape[:-2])

    # compute sum(w_i log(w_i)) where w_i are rho's eigenvalues
    w = jnp.linalg.eigvalsh(x)
    # we set small negative or null eigenvalues to 1.0 to avoid `nan` propagation
    w = jnp.where(w <= 0, 1.0, w)
    return -(w * jnp.log(w)).sum(-1)
