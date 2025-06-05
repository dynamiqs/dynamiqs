from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._checks import check_type_int
from .._utils import cdtype
from ..qarrays.qarray import QArray
from ..qarrays.utils import asqarray
from .general import tensor
from .operators import displace

__all__ = [
    'basis',
    'basis_dm',
    'coherent',
    'coherent_dm',
    'excited',
    'excited_dm',
    'fock',
    'fock_dm',
    'ground',
    'ground_dm',
    'thermal_dm',
]


def fock(dim: int | tuple[int, ...], number: ArrayLike) -> QArray:
    r"""Returns the ket of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array-like of shape (...) or (..., len(dim)))_: Fock state number
            for each mode, of integer type. If `dim` is a tuple, the last dimension of
            `number` should match the length of `dim`.

    Returns:
        _(qarray of shape (..., n, 1))_ Ket of the Fock state or tensor product of Fock
            states, with _n = prod(dims)_.

    Examples:
        Single-mode Fock state $\ket{1}$:
        >>> dq.fock(3, 1)
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]

        Batched single-mode Fock states $\{\ket{0}\!, \ket{1}\!, \ket{2}\}$:
        >>> dq.fock(3, [0, 1, 2])
        QArray: shape=(3, 3, 1), dims=(3,), dtype=complex64, layout=dense
        [[[1.+0.j]
          [0.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [1.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [0.+0.j]
          [1.+0.j]]]

        Multi-mode Fock state $\ket{1,0}$:
        >>> dq.fock((3, 2), (1, 0))
        QArray: shape=(6, 1), dims=(3, 2), dtype=complex64, layout=dense
        [[0.+0.j]
         [0.+0.j]
         [1.+0.j]
         [0.+0.j]
         [0.+0.j]
         [0.+0.j]]

        Batched multi-mode Fock states $\{\ket{0,0}\!, \ket{0,1}\!, \ket{1,1}\!,
        \ket{2,0}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock((3, 2), number).shape
        (4, 6, 1)
    """
    dim = np.asarray(dim)
    number = jnp.asarray(number)
    check_type_int(dim, 'dim')
    check_type_int(number, 'number')

    # check if dim is a single value or a tuple
    if dim.ndim > 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # if dim is an integer, convert shapes dim: () -> (1,) and number: (...) -> (..., 1)
    if dim.ndim == 0:
        dim = dim[None]
        number = number[..., None]

    # check if number has shape (..., len(ndim))
    if number.shape[-1] != dim.shape[-1]:
        raise ValueError(
            'Argument `number` must have shape `(...)` or `(..., len(dim))`, but'
            f' has shape number.shape={number.shape}.'
        )

    # check if 0 <= number[..., i] < dim[i] for all i
    number = eqx.error_if(
        number,
        dim - number <= 0,
        'Argument `number` must be in the range [0, dim[i]) for each mode i:'
        ' 0 <= number[..., i] < dim[i].',
    )

    # compute all kets
    def _fock(number: Array) -> QArray:
        # return the tensor product of Fock states |n0> x |n1> x ... x |nf> where dim
        # has shape (ndim,), number has shape (ndim,) and number = [n0, n1,..., nf]
        # this is the unbatched version of fock()
        idx = 0
        for d, n in zip(dim, number, strict=True):
            idx = d * idx + n
        ket = jnp.zeros((prod(dim), 1), dtype=cdtype())
        array = ket.at[idx].set(1.0)
        return asqarray(array, dims=tuple(dim.tolist()))

    _vectorized_fock = jnp.vectorize(_fock, signature='(ndim)->(prod_ndim,1)')
    return _vectorized_fock(number)


def fock_dm(dim: int | tuple[int, ...], number: ArrayLike) -> QArray:
    r"""Returns the density matrix of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array-like of shape (...) or (..., len(dim)))_: Fock state number
            for each mode, of integer type. If `dim` is a tuple, the last dimension of
            `number` should match the length of `dim`.

    Returns:
        _(qarray of shape (..., n, n))_ Density matrix of the Fock state or tensor
            product of Fock states, with _n = prod(dims)_.

    Examples:
        Single-mode Fock state $\ket{1}\bra{1}$:
        >>> dq.fock_dm(3, 1)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]]

        Batched single-mode Fock states $\{\ket{0}\bra{0}\!, \ket{1}\bra{1}\!,
        \ket{2}\bra{2}\}$:
        >>> dq.fock_dm(3, [0, 1, 2])
        QArray: shape=(3, 3, 3), dims=(3,), dtype=complex64, layout=dense
        [[[1.+0.j 0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j 0.+0.j]]
        <BLANKLINE>
         [[0.+0.j 0.+0.j 0.+0.j]
          [0.+0.j 1.+0.j 0.+0.j]
          [0.+0.j 0.+0.j 0.+0.j]]
        <BLANKLINE>
         [[0.+0.j 0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j 1.+0.j]]]

        Multi-mode Fock state $\ket{1,0}\bra{1,0}$:
        >>> dq.fock_dm((3, 2), (1, 0))
        QArray: shape=(6, 6), dims=(3, 2), dtype=complex64, layout=dense
        [[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

        Batched multi-mode Fock states $\{\ket{0,0}\bra{0,0}\!, \ket{0,1}\bra{0,1}\!,
        \ket{1,1}\bra{1,1}\!, \ket{2,0}\bra{2,0}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock_dm((3, 2), number).shape
        (4, 6, 6)
    """
    return fock(dim, number).todm()


def basis(dim: int | tuple[int, ...], number: ArrayLike) -> QArray:
    """Alias of [`dq.fock()`][dynamiqs.fock]."""
    return fock(dim, number)


def basis_dm(dim: int | tuple[int, ...], number: ArrayLike) -> QArray:
    """Alias of [`dq.fock_dm()`][dynamiqs.fock_dm]."""
    return fock_dm(dim, number)


def coherent(dim: int | tuple[int, ...], alpha: ArrayLike | list[ArrayLike]) -> QArray:
    r"""Returns the ket of a coherent state or a tensor product of coherent states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array-like of shape (...) or (len(dim), ...))_: Coherent state
            amplitude for each mode. If `dim` is a tuple, the first dimension of
            `alpha` should match the length of `dim`.

    Note:
        If you provide argument `alpha` as a list, all elements must be broadcastable.

    Returns:
        _(qarray of shape (..., n, 1))_ Ket of the coherent state or tensor product of
            coherent states, with _n = prod(dims)_.

    Examples:
        Single-mode coherent state $\ket{\alpha}$:
        >>> dq.coherent(4, 0.5)
        QArray: shape=(4, 1), dims=(4,), dtype=complex64, layout=dense
        [[0.882+0.j]
         [0.441+0.j]
         [0.156+0.j]
         [0.047+0.j]]

        Batched single-mode coherent states $\{\ket{\alpha_0}\!, \ket{\alpha_1}\}$:
        >>> dq.coherent(4, [0.5, 0.5j])
        QArray: shape=(2, 4, 1), dims=(4,), dtype=complex64, layout=dense
        [[[ 0.882+0.j   ]
          [ 0.441+0.j   ]
          [ 0.156+0.j   ]
          [ 0.047+0.j   ]]
        <BLANKLINE>
         [[ 0.882+0.j   ]
          [ 0.   +0.441j]
          [-0.156+0.j   ]
          [ 0.   -0.047j]]]


        Multi-mode coherent state $\ket{\alpha}\otimes\ket{\beta}$:
        >>> dq.coherent((2, 3), (0.5, 0.5j))
        QArray: shape=(6, 1), dims=(2, 3), dtype=complex64, layout=dense
        [[ 0.775+0.j   ]
         [ 0.   +0.386j]
         [-0.146+0.j   ]
         [ 0.423+0.j   ]
         [ 0.   +0.211j]
         [-0.08 +0.j   ]]

        Batched multi-mode coherent states $\{\ket{\alpha_0}\otimes\ket{\beta_0}\!,
        \ket{\alpha_1}\otimes\ket{\beta_1}\}$:
        >>> alpha1 = np.linspace(0, 1, 5)
        >>> alpha2 = np.linspace(0, 1, 7)
        >>> dq.coherent((8, 8), (alpha1[None, :], alpha2[:, None])).shape
        (7, 5, 64, 1)
    """
    dim = np.asarray(dim)
    check_type_int(dim, 'dim')

    # check if dim is a single value or a tuple
    if dim.ndim > 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # tackle multi-modes
    if dim.ndim == 1:
        return tensor(*[coherent(d, a) for d, a in zip(dim, alpha, strict=True)])

    # fact: dim is now an integer
    return displace(int(dim), alpha) @ fock(int(dim), 0)


def coherent_dm(dim: int | tuple[int, ...], alpha: ArrayLike) -> QArray:
    r"""Returns the density matrix of a coherent state or a tensor product of coherent
    states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array-like of shape (...) or (..., len(dim)))_: Coherent state
            amplitude for each mode. If `dim` is a tuple, the last dimension of
            `alpha` should match the length of `dim`.

    Returns:
        _(qarray of shape (..., n, n))_ Density matrix of the coherent state or tensor
            product of coherent states, with _n = prod(dims)_.

    Examples:
        Single-mode coherent state $\ket{\alpha}\bra{\alpha}$:
        >>> dq.coherent_dm(4, 0.5)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[0.779+0.j 0.389+0.j 0.137+0.j 0.042+0.j]
         [0.389+0.j 0.195+0.j 0.069+0.j 0.021+0.j]
         [0.137+0.j 0.069+0.j 0.024+0.j 0.007+0.j]
         [0.042+0.j 0.021+0.j 0.007+0.j 0.002+0.j]]

        Batched single-mode coherent states $\{\ket{\alpha_0}\bra{\alpha_0}\!,
        \ket{\alpha_1}\bra{\alpha_1}\}$:
        >>> dq.coherent_dm(4, [0.5, 0.5j]).shape
        (2, 4, 4)

        Multi-mode coherent state
        $\ket{\alpha}\bra{\alpha}\otimes\ket{\beta}\bra{\beta}$:
        >>> dq.coherent_dm((2, 3), (0.5, 0.5j)).shape
        (6, 6)

        Batched multi-mode coherent states
        $\{\ket{\alpha_0}\bra{\alpha_0}\otimes\ket{\beta_0}\bra{\beta_0}\!,
        \ket{\alpha_1}\bra{\alpha_1}\otimes\ket{\beta_1}\bra{\beta_1}\}$:
        >>> alpha = [(0.5, 0.5j), (0.5j, 0.5)]
        >>> dq.coherent_dm((4, 6), alpha).shape
        (2, 24, 24)
    """
    return coherent(dim, alpha).todm()


def ground() -> QArray:
    r"""Returns the eigenvector with eigenvalue $-1$ of the Pauli $\sigma_z$ operator.

    It is defined by $\ket{g} = \begin{pmatrix}0\\1\end{pmatrix}$.

    Note:
        This function is named `ground` because $\ket{g}$ is the lower energy state of
        a two-level system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(qarray of shape (2, 1))_ Ket $\ket{g}$.

    Examples:
        >>> dq.ground()
        QArray: shape=(2, 1), dims=(2,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]]
    """
    return asqarray(jnp.array([[0], [1]], dtype=cdtype()), dims=(2,))


def ground_dm() -> QArray:
    r"""Returns the projector on the eigenvector with eigenvalue $-1$ of the Pauli
    $\sigma_z$ operator.

    It is defined by $\ket{g}\bra{g} = \begin{pmatrix}0 & 0\\0 & 1\end{pmatrix}$.

    Note:
        This function is named `ground_dm` because $\ket{g}$ is the lower energy state
        of a two-level system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(qarray of shape (2, 2))_ Density matrix $\ket{g}\bra{g}$.

    Examples:
        >>> dq.ground_dm()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j]]
    """
    return ground().todm()


def excited() -> QArray:
    r"""Returns the eigenvector with eigenvalue $+1$ of the Pauli $\sigma_z$ operator.

    It is defined by $\ket{e} = \begin{pmatrix}1\\0\end{pmatrix}$.

    Note:
        This function is named `excited` because $\ket{e}$ is the higher energy state of
        a two-level-system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(qarray of shape (2, 1))_ Ket $\ket{e}$.

    Examples:
        >>> dq.excited()
        QArray: shape=(2, 1), dims=(2,), dtype=complex64, layout=dense
        [[1.+0.j]
         [0.+0.j]]
    """
    return asqarray(jnp.array([[1], [0]], dtype=cdtype()), dims=(2,))


def excited_dm() -> QArray:
    r"""Returns the projector on the eigenvector with eigenvalue $+1$ of the Pauli
    $\sigma_z$ operator.

    It is defined by $\ket{e}\bra{e} = \begin{pmatrix}1 & 0\\0 & 0\end{pmatrix}$.

    Note:
        This function is named `excited_dm` because $\ket{e}$ is the higher energy state
        of a two-level-system with Hamiltonian $H=\omega \sigma_z$.

    Returns:
        _(qarray of shape (2, 2))_ Density matrix $\ket{e}\bra{e}$.

    Examples:
        >>> dq.excited_dm()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j]]
    """
    return excited().todm()


def thermal_dm(dim: int | tuple[int, ...], nth: ArrayLike) -> QArray:
    r"""Returns the density matrix of a thermal state or a tensor product of thermal
    states.

    For a single mode, it is defined for a thermal photon number $n_{th}$ by:

    $$
        \rho = \sum_k \frac{(n_{th})^k}{(1+n_{th})^{1+k}} \ket{k}\bra{k}.
    $$

    Args:
        dim: Hilbert space dimension of each mode.
        nth _(array-like of shape (...) or (..., len(dim)))_: Thermal photon number for
            each mode. If `dim` is a tuple, the last dimension of `nth` should match the
            length of `dim`.

    Returns:
        _(qarray of shape (..., n, n))_ Density matrix of the thermal state or tensor
            product of thermal states, with _n = prod(dims)_.

    Examples:
        Single-mode thermal state with thermal photon number $n_{th}=0.1$:
        >>> dq.thermal_dm(4, 0.1)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[0.909+0.j 0.   +0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.083+0.j 0.   +0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.008+0.j 0.   +0.j]
         [0.   +0.j 0.   +0.j 0.   +0.j 0.001+0.j]]

        Batched single-mode thermal states:
        >>> dq.thermal_dm(4, [0.1, 0.2, 0.3]).shape
        (3, 4, 4)

        Multi-mode thermal state:
        >>> dq.thermal_dm((4, 3), (0.1, 0.2)).shape
        (12, 12)

        Batched multi-mode thermal states:
        >>> nth = [(0.1, 0.2), (0.2, 0.1), (0.2, 0.2)]
        >>> dq.thermal_dm((4, 3), nth).shape
        (3, 12, 12)
    """
    dim = np.asarray(dim)
    nth = jnp.asarray(nth)
    check_type_int(dim, 'dim')

    # check if dim is a single value or a tuple
    if dim.ndim > 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # if dim is an integer, convert shapes dim: () -> (1,) and nth: (...) -> (..., 1)
    if dim.ndim == 0:
        dim = dim[None]
        nth = nth[..., None]

    # check if nth has shape (..., len(ndim))
    if nth.shape[-1] != dim.shape[-1]:
        raise ValueError(
            'Argument `nth` must have shape `(...)` or `(..., len(dim))`, but'
            f' has shape nth.shape={nth.shape}.'
        )

    # compute all density matrices
    def _thermal_dm(nth: Array) -> QArray:
        dms = [_single_thermal_dm(d, n) for d, n in zip(dim, nth, strict=True)]
        return tensor(*dms)

    _vectorized_thermal_dm = jnp.vectorize(
        _thermal_dm, signature='(ndim)->(prod_ndim,prod_ndim)'
    )
    return _vectorized_thermal_dm(nth)


def _single_thermal_dm(dim: int, nth: Array) -> QArray:
    """Returns the density matrix of a thermal state for a single mode."""
    # compute the unnormalized diagonal elements of the density matrix
    fock_indices = jnp.arange(dim)
    rho_diag = (nth**fock_indices) / ((1 + nth) ** (1 + fock_indices))

    # cast to complex dtype
    rho_diag = rho_diag.astype(cdtype())

    # construct the density matrix
    bdiag = jnp.vectorize(jnp.diag, signature='(a)->(a,a)')
    rho = asqarray(bdiag(rho_diag), dims=(dim.item(),))

    # normalize the density matrix
    return rho.unit()
