from __future__ import annotations

from functools import partial, wraps
from math import sqrt
from typing import Iterator

import torch
from methodtools import lru_cache
from torch import Tensor
from tqdm import tqdm as std_tqdm

from ...utils.utils import isket

# define a default progress bar format
PBAR_FORMAT = '|{bar}| {percentage:4.1f}% - time {elapsed}/{remaining}'

# redefine tqdm with some default arguments
tqdm = partial(std_tqdm, bar_format=PBAR_FORMAT)


def inv_sqrtm(mat: Tensor) -> Tensor:
    """Compute the inverse square root of a complex Hermitian or real symmetric matrix
    using its eigendecomposition.

    TODO Replace with Schur decomposition once released by PyTorch.
         See the feature request at https://github.com/pytorch/pytorch/issues/78809.
         Alternatively, see a sqrtm implementation at
         https://github.com/pytorch/pytorch/issues/25481#issuecomment-584896176.
    """
    vals, vecs = torch.linalg.eigh(mat)
    inv_sqrt_vals = torch.diag_embed(vals ** (-0.5)).to(vecs)
    return vecs @ torch.linalg.solve(vecs, inv_sqrt_vals, left=False)


def bexpect(O: Tensor, x: Tensor) -> Tensor:
    r"""Compute the expectation values of batched operators on a state vector or a
    density matrix.

    The expectation value $\braket{O}$ of a single operator $O$ is computed
    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a state vector $\psi$,
    - as $\braket{O}=\tr(O\rho)$ if `x` is a density matrix $\rho$.

    Notes:
        The returned tensor is complex-valued.

    TODO Adapt to bras.

    Args:
        O: Tensor of shape `(b, n, n)`.
        x: Tensor of shape `(..., n, 1)` or `(..., n, n)`.

    Returns:
        Tensor of shape `(..., b)` holding the operators expectation values.
    """
    if isket(x):
        return torch.einsum('...ij,bjk,...kl->...b', x.mH, O, x)  # <x|O|x>
    return torch.einsum('bij,...ji->...b', O, x)  # tr(Ox)


def none_to_zeros_like(
    in_tuple: tuple[Tensor | None, ...], shaping_tuple: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    """Convert `None` values of `in_tuple` to zero-valued tensors with the same shape
    as `shaping_tuple`."""
    return tuple(
        torch.zeros_like(s) if a is None else a for a, s in zip(in_tuple, shaping_tuple)
    )


def add_tuples(a: tuple, b: tuple) -> tuple:
    """Element-wise sum of two tuples of the same shape."""
    return tuple(x + y for x, y in zip(a, b))


def hairer_norm(x: Tensor) -> Tensor:
    """Rescaled Frobenius norm of a batched matrix.

    See Equation (4.11) of `Hairer et al., Solving Ordinary Differential Equations I
    (1993), Springer Series in Computational Mathematics`.

    Args:
        x: Tensor of shape `(..., n, n)`.

    Returns:
        Tensor of shape `(...)` holding the norm of each matrix in the batch.
    """
    return torch.linalg.matrix_norm(x) / sqrt(x.size(-1) * x.size(-2))


def cache(func=None, *, maxsize: int = 1):
    """Cache a function returning a tensor by memoizing its most recent calls.

    This decorator extends `methodtools.lru_cache` to also cache a function on
    PyTorch grad mode status (enabled or disabled). This prevents cached tensors
    detached from the graph (for example computed within a `with torch.no_grad()`
    block) from being used by mistake by later code which requires tensors attached
    to the graph.

    By default, the cache size is `1`, which means that only the most recent call is
    cached. Use the `maxsize` keyword argument to change the maximum cache size.

    Warning:
        This decorator should only be used for PyTorch tensors.

    Example:
        >>> @cache
        ... def square(x: Tensor) -> Tensor:
        ...     print('compute square')
        ...     return x**2
        ...
        >>> x = torch.tensor([1, 2, 3])
        >>> square(x)
        compute square
        tensor([1, 4, 9])
        >>> square(x)
        tensor([1, 4, 9])
        >>> with torch.no_grad():
        ...     print(square(x))
        ...     print(square(x))
        ...
        compute square
        tensor([1, 4, 9])
        tensor([1, 4, 9])

        Increasing the maximum cache size:
        >>> @cache(maxsize=2)
        ... def square(x):
        ...     print('compute square')
        ...     return x**2
        ...
        >>> square(1)
        compute square
        1
        >>> square(2)
        compute square
        4
        >>> square(1)
        1
        >>> square(2)
        4
        >>> square(3)
        compute square
        9
        >>> square(2)
        4
        >>> square(1)
        compute square
        1

    Args:
        func: Function returning a tensor, can take any number of arguments.

    Returns:
        Cached function.
    """
    if func is None:
        return partial(cache, maxsize=maxsize)

    # define a function cached on its arguments and also on PyTorch grad mode status
    @lru_cache(maxsize=maxsize)
    def grad_cached_func(*args, grad_enabled, **kwargs):
        return func(*args, **kwargs)

    # wrap `func` to call its modified cached version
    @wraps(func)
    def wrapper(*args, **kwargs):
        return grad_cached_func(*args, grad_enabled=torch.is_grad_enabled(), **kwargs)

    return wrapper


def iteraxis(x: Tensor, axis: int = 0) -> Iterator[Tensor]:
    for i in range(x.size(axis)):
        yield x.select(axis, i)


def format_L(L: list[Tensor]) -> Tensor:
    # Format a list of tensors of individual shape (n, n) or (?, n, n) into a single
    # batched tensor of shape (nL, bL, n, n). An error is raised if all batched
    # dimensions `?` are not the same.

    n = L[0].size(-1)
    L = [x.view(-1, n, n) for x in L]  # [(?, n, n)] with ? = 1 if not batched
    bs = torch.tensor([x.size(0) for x in L])  # list of batch sizes (the ?)

    # get the unique batch size or raise an error if batched dimensions are not the same
    bs_unique = torch.unique(bs)
    bs_unique_not_one = bs_unique[bs_unique != 1]
    if len(bs_unique_not_one) > 1:
        L_shapes = [tuple(x.shape) for x in L]
        raise ValueError(
            'Argument `jump_ops` should be a list of 2D arrays or 3D arrays with the'
            ' same batch size, but got a list of arrays with incompatible shapes'
            f' {L_shapes}.'
        )
    elif len(bs_unique_not_one) == 1:
        bL = bs_unique_not_one.item()
    else:
        bL = 1

    L = [x.repeat((bL if s == 1 else 1), 1, 1) for x, s in zip(L, bs)]
    return torch.stack(L)
