from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from .._utils import cdtype
from .operators import displace
from .utils import todm

__all__ = ['fock', 'fock_dm', 'basis', 'basis_dm', 'coherent', 'coherent_dm']


def fock(dim: int, number: ArrayLike) -> Array:
    r"""Returns the ket of a Fock state.

    Args:
        dim: Dimension of the Hilbert space.
        number _(array-like of integers of shape (...))_: Fock state number.

    Returns:
        _(array of shape (..., dim, 1))_ Ket of the Fock state, or array of kets of
            Fock states if batched.

    Examples:
        >>> dq.fock(3, 1)
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.fock(3, (1, 0))
        Array([[[0.+0.j],
                [1.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[1.+0.j],
                [0.+0.j],
                [0.+0.j]]], dtype=complex64)
    """
    # check `dim` is an integer
    dim = jnp.asarray(dim)
    if not jnp.issubdtype(dim.dtype, jnp.integer) and dim.ndim == 0:
        raise TypeError('Argument `dim` must be an integer.')

    # check if number is an integer or an array-like of integers
    number = jnp.asarray(number)
    if not jnp.issubdtype(number.dtype, jnp.integer):
        raise TypeError(
            'Argument `number` must be an integer, or an array-like of' 'integers.'
        )

    # check if all numbers are within [0, dim)
    if jnp.any(jnp.logical_or(number < 0, number >= dim)):
        raise ValueError('Fock state number must be in the range [0, dim).')

    # compute all kets
    kets = jnp.zeros((*number.shape, dim, 1), dtype=cdtype())
    for idx, n in np.ndenumerate(number):
        kets = kets.at[(*idx, n, 0)].set(1.0)
    return kets


def fock_dm(dim: int, number: ArrayLike) -> Array:
    r"""Returns the density matrix of a Fock state.

    Args:
        dim: Dimension of the Hilbert space.
        number _(array-like of integers of shape (...))_: Fock state number.

    Returns:
        _(array of shape (..., dim, dim))_ Density matrix of the Fock state, or array of
            density matrices of Fock states if batched.

    Examples:
        >>> dq.fock_dm(3, 1)
        Array([[0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        >>> dq.fock_dm(3, (1, 0))
        Array([[[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
               [[1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]]], dtype=complex64)
    """
    return todm(fock(dim, number))


def basis(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Array:
    """Alias of [`dq.fock()`][dynamiqs.fock]."""
    return fock(dim, number)


def basis_dm(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Array:
    """Alias of [`dq.fock_dm()`][dynamiqs.fock_dm]."""
    return fock_dm(dim, number)


def coherent(dim: int, alpha: ArrayLike) -> Array:
    r"""Returns the ket of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(array-like of shape (...))_: Coherent state amplitude.

    Returns:
        _(array of shape (..., dim, 1))_ Ket of the coherent state, or array of kets of
            coherent states if batched.

    Examples:
        >>> dq.coherent(4, 0.5)
        Array([[0.882+0.j],
               [0.441+0.j],
               [0.156+0.j],
               [0.047+0.j]], dtype=complex64)
        >>> dq.coherent(4, (0.5, 0.5j))
        Array([[[ 0.882+0.j   ],
                [ 0.441+0.j   ],
                [ 0.156+0.j   ],
                [ 0.047+0.j   ]],
        <BLANKLINE>
               [[ 0.882+0.j   ],
                [ 0.   +0.441j],
                [-0.156+0.j   ],
                [ 0.   -0.047j]]], dtype=complex64)
    """
    # check `dim` is an integer
    dim = jnp.asarray(dim)
    if not jnp.issubdtype(dim.dtype, jnp.integer) and dim.ndim == 0:
        raise TypeError('Argument `dim` must be an integer.')

    # compute all kets
    return displace(dim, alpha) @ fock(dim, 0)


def coherent_dm(dim: int, alpha: ArrayLike) -> Array:
    r"""Returns the density matrix of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(array-like of shape (...))_: Coherent state amplitude.

    Returns:
        _(array of shape (..., dim, dim))_ Density matrix of the coherent state, or
            array of density matrices of coherent states if batched.

    Examples:
        >>> dq.coherent_dm(4, 0.5)
        Array([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
               [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
               [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
               [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]], dtype=complex64)
        >>> dq.coherent_dm(4, (0.5, 0.5j))
        Array([[[ 0.779+0.j   ,  0.389+0.j   ,  0.137+0.j   ,  0.042+0.j   ],
                [ 0.389+0.j   ,  0.195+0.j   ,  0.069+0.j   ,  0.021+0.j   ],
                [ 0.137+0.j   ,  0.069+0.j   ,  0.024+0.j   ,  0.007+0.j   ],
                [ 0.042+0.j   ,  0.021+0.j   ,  0.007+0.j   ,  0.002+0.j   ]],
        <BLANKLINE>
               [[ 0.779+0.j   ,  0.   -0.389j, -0.137-0.j   ,  0.   +0.042j],
                [ 0.   +0.389j,  0.195+0.j   ,  0.   -0.069j, -0.021+0.j   ],
                [-0.137+0.j   ,  0.   +0.069j,  0.024+0.j   , -0.   -0.007j],
                [ 0.   -0.042j, -0.021-0.j   , -0.   +0.007j,  0.002+0.j   ]]],      dtype=complex64)
    """  # noqa: E501
    return todm(coherent(dim, alpha))
