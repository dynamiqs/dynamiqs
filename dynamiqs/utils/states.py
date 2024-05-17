from __future__ import annotations

from math import prod

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._utils import cdtype
from .operators import displace
from .utils import tensor, todm

__all__ = ['fock', 'fock_dm', 'basis', 'basis_dm', 'coherent', 'coherent_dm']


def fock(dim: int | tuple[int, ...], number: ArrayLike) -> Array:
    r"""Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dim: Dimension of the Hilbert space of each mode.
        number _(integer array-like of shape (...) or (..., len(dim)))_): Fock state
            number of each mode, with last dimension matching the length of `dim` if
            `dim` is a tuple.

    Returns:
        _(array of shape (..., prod(dim), 1))_ Ket of the Fock state or
            tensor product of Fock states.

    Examples:
        Single-mode Fock states:
        >>> dq.fock(3, 1)
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)

        Multi-mode Fock states:
        >>> dq.fock((3, 2), (1, 0))
        Array([[0.+0.j],
               [0.+0.j],
               [1.+0.j],
               [0.+0.j],
               [0.+0.j],
               [0.+0.j]], dtype=complex64)

        Batched single-mode Fock states:
        >>> number = jnp.array([0, 1, 2])
        >>> dq.fock(3, number)
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

        Batched multi-mode Fock states:
        >>> number = [(0, 0), (0, 1), (1, 1), (2, 0)]
        >>> dq.fock((3, 2), number)
        Array([[[1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j]]], dtype=complex64)
    """
    # check if dim is an integer or tuple of integers
    dim = jnp.asarray(dim)
    dim = dim[None] if dim.ndim == 0 else dim
    if not jnp.issubdtype(dim.dtype, jnp.integer) and not dim.ndim == 1:
        raise ValueError('Argument `dim` must be an integer or a tuple of integers.')

    # check if number is an integer array-like object
    number = jnp.asarray(number)
    number = number[None] if number.ndim == 0 else number
    number = number[..., None] if len(dim) == 1 and number.shape[-1] != 1 else number
    if not jnp.issubdtype(number.dtype, jnp.integer):
        raise ValueError('Argument `number` must be an integer array-like object.')

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


def fock_dm(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Array:
    r"""Returns the density matrix of a Fock state or the density matrix of a tensor
    product of Fock states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        number _(int or tuple of ints)_: Fock state number of each mode.

    Returns:
        _(array of shape (n, n))_ Density matrix of the Fock state or tensor product of
            Fock states.

    Examples:
        >>> dq.fock_dm(3, 1)
        Array([[0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        >>> dq.fock_dm((3, 2), (1, 0))
        Array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    """
    return todm(fock(dim, number))


def basis(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Array:
    """Alias of [`dq.fock()`][dynamiqs.fock]."""
    return fock(dim, number)


def basis_dm(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Array:
    """Alias of [`dq.fock_dm()`][dynamiqs.fock_dm]."""
    return fock_dm(dim, number)


def coherent(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the ket of a coherent state, or the ket of a tensor product of coherent
    states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alpha _(array_like)_: Coherent state amplitude of each mode.

    Returns:
        _(array of shape (n, 1))_ Ket of the coherent state.

    Examples:
        >>> dq.coherent(4, 0.5)
        Array([[0.882+0.j],
               [0.441+0.j],
               [0.156+0.j],
               [0.047+0.j]], dtype=complex64)
        >>> dq.coherent((2, 3), (0.5, 0.5j))
        Array([[ 0.775+0.j   ],
               [ 0.   +0.386j],
               [-0.146+0.j   ],
               [ 0.423+0.j   ],
               [ 0.   +0.211j],
               [-0.08 +0.j   ]], dtype=complex64)
    """
    # convert inputs to tuples by default, and check dimensions match
    dim = (dim,) if isinstance(dim, int) else dim
    alpha = jnp.asarray(alpha, dtype=cdtype())

    if alpha.ndim == 0:
        alpha = alpha[..., None]
    elif alpha.ndim > 1:
        raise ValueError(
            'Argument `alpha` must be a 0-D or 1-D array-like object, but is'
            f' a {alpha.ndim}-D object.'
        )
    if len(dim) != len(alpha):
        raise ValueError(
            'Arguments `alpha` must have the same length as `dim` of length'
            f' {len(dim)}, but has length {len(alpha)}.'
        )

    kets = [displace(d, a) @ fock(d, 0) for d, a in zip(dim, alpha)]
    return tensor(*kets)


def coherent_dm(dim: int | tuple[int, ...], alpha: ArrayLike) -> Array:
    r"""Returns the density matrix of a coherent state, or the density matrix of a
    tensor product of coherent states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        alpha _(array_like)_: Coherent state amplitude of each mode.

    Returns:
        _(array of shape (n, n))_ Density matrix of the coherent state.

    Examples:
        >>> dq.coherent_dm(4, 0.5)
        Array([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
               [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
               [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
               [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]], dtype=complex64)
        >>> dq.coherent_dm((2, 3), (0.5, 0.5))
        Array([[0.6  +0.j, 0.299+0.j, 0.113+0.j, 0.328+0.j, 0.163+0.j, 0.062+0.j],
               [0.299+0.j, 0.149+0.j, 0.056+0.j, 0.163+0.j, 0.081+0.j, 0.031+0.j],
               [0.113+0.j, 0.056+0.j, 0.021+0.j, 0.062+0.j, 0.031+0.j, 0.012+0.j],
               [0.328+0.j, 0.163+0.j, 0.062+0.j, 0.179+0.j, 0.089+0.j, 0.034+0.j],
               [0.163+0.j, 0.081+0.j, 0.031+0.j, 0.089+0.j, 0.044+0.j, 0.017+0.j],
               [0.062+0.j, 0.031+0.j, 0.012+0.j, 0.034+0.j, 0.017+0.j, 0.006+0.j]],      dtype=complex64)
    """  # noqa: E501
    return todm(coherent(dim, alpha))
