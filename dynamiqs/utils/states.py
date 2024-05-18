from __future__ import annotations

from math import prod

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._checks import check_type_int
from .._utils import cdtype
from .operators import displace
from .utils import tensor, todm

__all__ = ['fock', 'fock_dm', 'basis', 'basis_dm', 'coherent', 'coherent_dm']


def fock(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    r"""Returns the ket of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array_like of shape (...) or (..., len(dim)))_: Number of particles
            for each mode, of integer type.

    Returns:
        _(array of shape (..., prod(dim), 1))_ Ket of the Fock state or tensor product
            of Fock states.

    Examples:
        Single-mode Fock state $\ket{1}$:
        >>> dq.fock(3, 1)
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)

        Batched single-mode Fock states $\{\ket{0}, \ket{1}, \ket{2}\}$:
        >>> dq.fock(3, [0, 1, 2])
        Array([[[1.+0.j],
                [0.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [1.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [0.+0.j],
                [1.+0.j]]], dtype=complex64)

        Multi-mode Fock state $\ket{10}$:
        >>> dq.fock((3, 2), (1, 0))
        Array([[0.+0.j],
               [0.+0.j],
               [1.+0.j],
               [0.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)

        Batched multi-mode Fock states $\{\ket{00}, \ket{01}, \ket{11}, \ket{20}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock((3, 2), number).shape
        (4, 6, 1)
    """
    # check if dim is an integer or tuple of integers
    dim = jnp.asarray(dim)
    dim = dim[None] if dim.ndim == 0 else dim
    check_type_int(dim, 'dim')
    if not dim.ndim == 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # check if number is an integer array-like object
    number = jnp.asarray(number)
    number = number[None] if number.ndim == 0 else number
    number = number[..., None] if len(dim) == 1 and number.shape[-1] != 1 else number
    check_type_int(number, 'number')

    # check number and dim shapes match
    if dim.shape[-1] != number.shape[-1]:
        raise ValueError(
            'Arguments `number` and `dim` must have compatible shapes, but got'
            f' dim.shape={dim.shape}, and number.shape={number.shape}.'
        )

    # check if all numbers are within [0, dim)
    for i, d in enumerate(dim):
        if jnp.any(jnp.logical_or(number[..., i] < 0, number[..., i] >= d)):
            raise ValueError(
                'Fock state number must be in the range [0, dim) for each mode.'
            )

    # compute all kets
    number = number.swapaxes(0, -1)  # (len(dim), ...)
    batch_shape = number.shape[1:]
    kets = jnp.zeros((*batch_shape, prod(dim), 1), dtype=cdtype())
    indices = jnp.zeros(batch_shape, dtype=jnp.int32)
    for d, ns in zip(dim, number):
        indices = d * indices + ns

    for batch_idx, index in np.ndenumerate(indices):
        kets = kets.at[(*batch_idx, index, 0)].set(1.0)

    return kets


def fock_dm(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    r"""Returns the density matrix of a Fock state or a tensor product of Fock states.

    Args:
        dim: Hilbert space dimension of each mode.
        number _(array_like of shape (...) or (..., len(dim)))_: Number of particles
            for each mode, of integer type.

    Returns:
        _(array of shape (..., prod(dim), prod(dim)))_ Density matrix of the Fock state
            or tensor product of Fock states.

    Examples:
        Single-mode Fock state $\ket{1}$:
        >>> dq.fock_dm(3, 1)
        Array([[0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)

        Batched single-mode Fock states $\{\ket{0}, \ket{1}, \ket{2}\}$:
        >>> dq.fock_dm(3, [0, 1, 2])
        Array([[[1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
               [[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]],
        <BLANKLINE>
               [[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j]]], dtype=complex64)

        Multi-mode Fock state $\ket{10}$:
        >>> dq.fock_dm((3, 2), (1, 0))
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)

        Batched multi-mode Fock states $\{\ket{00}, \ket{01}, \ket{11}, \ket{20}\}$:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock_dm((3, 2), number).shape
        (4, 6, 6)
    """
    return todm(fock(dim, number))


def basis(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    """Alias of [`dq.fock()`][dynamiqs.fock]."""
    return fock(dim, number)


def basis_dm(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    """Alias of [`dq.fock_dm()`][dynamiqs.fock_dm]."""
    return fock_dm(dim, number)


def coherent(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the ket of a coherent state or a tensor product of coherent states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array_like of shape (...) or (..., len(dim)))_: Coherent state
            amplitude for each mode.

    Returns:
        _(array of shape (..., prod(dim), 1))_ Ket of the coherent state or tensor
            product of coherent states.

    Examples:
        Single-mode coherent state $\ket{0.5}$:
        >>> dq.coherent(4, 0.5)
        Array([[0.882+0.j],
               [0.441+0.j],
               [0.156+0.j],
               [0.047+0.j]], dtype=complex64)

        Batched single-mode coherent states $\{\ket{0.5}, \ket{0.5i}\}$:
        >>> dq.coherent(4, [0.5, 0.5j])
        Array([[[ 0.882+0.j   ],
                [ 0.441+0.j   ],
                [ 0.156+0.j   ],
                [ 0.047+0.j   ]],
        <BLANKLINE>
               [[ 0.882+0.j   ],
                [ 0.   +0.441j],
                [-0.156+0.j   ],
                [ 0.   -0.047j]]], dtype=complex64)

        Multi-mode coherent state $\ket{0.5}\otimes\ket{0.5i}$:
        >>> dq.coherent((2, 3), (0.5, 0.5j))
        Array([[ 0.775+0.j   ],
               [ 0.   +0.386j],
               [-0.146+0.j   ],
               [ 0.423+0.j   ],
               [ 0.   +0.211j],
               [-0.08 +0.j   ]], dtype=complex64)

        Batched multi-mode coherent states $\{\ket{0.5}\otimes\ket{0.5i},
        \ket{0.5i}\otimes\ket{0.5}\}$:
        >>> alpha = [(0.5, 0.5j), (0.5j, 0.5)]
        >>> dq.coherent((4, 6), alpha).shape
        (2, 24, 1)
    """
    # check if dim is an integer or tuple of integers
    dim = jnp.asarray(dim)
    dim = dim[None] if dim.ndim == 0 else dim
    if not jnp.issubdtype(dim.dtype, jnp.integer) and not dim.ndim == 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # convert alpha to an array-like object of shape (..., len(dim))
    alpha = jnp.asarray(alpha)
    alpha = alpha[None] if alpha.ndim == 0 else alpha
    alpha = alpha[..., None] if len(dim) == 1 and alpha.shape[-1] != 1 else alpha

    # check alpha and dim shapes match
    if dim.shape[-1] != alpha.shape[-1]:
        raise ValueError(
            'Arguments `alpha` and `dim` must have compatible shapes, but got'
            f' dim.shape={dim.shape}, and alpha.shape={alpha.shape}.'
        )

    # compute all kets
    alpha = alpha.swapaxes(0, -1)  # (len(dim), ...)
    kets = [displace(d, a) @ fock(d, 0) for d, a in zip(dim, alpha)]
    return tensor(*kets)


def coherent_dm(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the density matrix of a coherent state or a tensor product of coherent
    states.

    Args:
        dim: Hilbert space dimension of each mode.
        alpha _(array_like of shape (...) or (..., len(dim)))_: Coherent state
            amplitude for each mode.

    Returns:
        _(array of shape (..., prod(dim), prod(dim)))_ Density matrix of the coherent
            state or tensor product of coherent states.

    Examples:
        Single-mode coherent state $\ket{0.5}$:
        >>> dq.coherent_dm(4, 0.5)
        Array([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
               [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
               [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
               [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]], dtype=complex64)

        Batched single-mode coherent states $\{\ket{0.5}, \ket{0.5i}\}$:
        >>> dq.coherent_dm(4, [0.5, 0.5j]).shape
        (2, 4, 4)

        Multi-mode coherent state $\ket{0.5}\otimes\ket{0.5i}$:
        >>> dq.coherent_dm((2, 3), (0.5, 0.5j)).shape
        (6, 6)

        Batched multi-mode coherent states $\{\ket{0.5}\otimes\ket{0.5i},
        \ket{0.5i}\otimes\ket{0.5}\}$:
        >>> alpha = [(0.5, 0.5j), (0.5j, 0.5)]
        >>> dq.coherent_dm((4, 6), alpha).shape
        (2, 24, 24)
    """
    return todm(coherent(dim, alpha))
