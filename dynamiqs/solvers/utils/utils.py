from __future__ import annotations

from functools import partial, wraps
from math import sqrt

import torch
from methodtools import lru_cache
from torch import Tensor
from tqdm import tqdm as std_tqdm

from ...utils.utils import is_ket

# define a default progress bar format
PBAR_FORMAT = '|{bar}| {percentage:4.1f}% - time {elapsed}/{remaining}'

# redefine tqdm with some default arguments
tqdm = partial(std_tqdm, bar_format=PBAR_FORMAT)


def kraus_map(rho: Tensor, O: Tensor) -> Tensor:
    """Compute the application of a Kraus map on an input density matrix.

    This is equivalent to `torch.sum(operators @ rho[None,...] @ operators.mH, dim=0)`.
    The use of einsum yields better performances on large matrices, but may cause a
    small overhead on smaller matrices (N <~ 50).

    TODO Fix documentation

    Args:
        rho: Density matrix of shape `(a, ..., n, n)`.
        operators: Kraus operators of shape `(a, b, n, n)`.

    Returns:
        Density matrix of shape `(a, ..., n, n)` with the Kraus map applied.
    """
    return torch.einsum('abij,a...jk,abkl->a...il', O, rho, O.mH)


def inv_sqrtm(mat: Tensor) -> Tensor:
    """Compute the inverse square root of a matrix using its eigendecomposition.

    TODO Replace with Schur decomposition once released by PyTorch.
         See the feature request at https://github.com/pytorch/pytorch/issues/78809.
         Alternatively, see a sqrtm implementation at
         https://github.com/pytorch/pytorch/issues/25481#issuecomment-584896176.
    """
    vals, vecs = torch.linalg.eigh(mat)
    inv_sqrt_vals = torch.diag(vals ** (-0.5)).to(vecs)
    return vecs @ torch.linalg.solve(vecs, inv_sqrt_vals, left=False)


def bexpect(O: Tensor, x: Tensor) -> Tensor:
    r"""Compute the expectation values of batched operators on a state vector or a
    density matrix.

    The expectation value $\braket{O}$ of a single operator $O$ is computed
    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a state vector $\psi$,
    - as $\braket{O}=\tr(O\rho)$ if `x` is a density matrix $\rho$.

    Note:
        The returned tensor is complex-valued.

    TODO Adapt to bras.

    Args:
        O: Tensor of size `(b, n, n)`.
        x: Tensor of size `(..., n, 1)` or `(..., n, n)`.

    Returns:
        Tensor of size `(..., b)` holding the operators expectation values.
    """
    if is_ket(x):
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
        x: Tensor of size `(..., n, n)`.

    Returns:
        Tensor of size `(...)` holding the norm of each matrix in the batch.
    """
    return torch.linalg.matrix_norm(x) / sqrt(x.size(-1) * x.size(-2))


def cache(func):
    """Cache a function returning a tensor by memoizing its most recent call.

    This decorator extends `methodtools.lru_cache` to also cache a function on PyTorch
    grad mode status (enabled or disabled). This prevents cached tensors detached from
    the graph (for example computed within a `with torch.no_grad()` block) from being
    used by mistake by later code which requires tensors attached to the graph.

    Warning:
        This decorator should only be used for PyTorch tensors.

    Example:
        >>> @cache
        ... def square(x: Tensor): Tensor
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

    Args:
        func: Function returning a tensor, can take any number of arguments.

    Returns:
        Cached function.
    """

    # define a function cached on its arguments and also on PyTorch grad mode status
    @lru_cache(maxsize=1)
    def grad_cached_func(*args, grad_enabled, **kwargs):
        return func(*args, **kwargs)

    # wrap `func` to call its modified cached version
    @wraps(func)
    def wrapper(*args, **kwargs):
        return grad_cached_func(*args, grad_enabled=torch.is_grad_enabled(), **kwargs)

    return wrapper
