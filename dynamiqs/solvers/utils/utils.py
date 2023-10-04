from __future__ import annotations

from functools import partial, wraps
from math import sqrt

import torch
from methodtools import lru_cache
from torch import Tensor
from tqdm import tqdm as std_tqdm

from ...utils.utils import isket

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


class InvSqrtm(torch.autograd.Function):
    """
    Computes the inverse square root of a matrix or batch of matrices.
    Original code is from https://github.com/KingJamesSong/FastDifferentiableMatSqrt
    The algorithm is described in https://arxiv.org/abs/2201.08663
    """

    pade_p = torch.tensor([1.0, -2.75, 2.75, -1.203125, 0.21484375, -0.0107421875])
    pade_q = torch.tensor([1.0, -2.25, 1.75, -0.546875, 0.05859375, -0.0009765625])

    @staticmethod
    def forward(ctx, mat):
        """
        Args:
            ctx: Pytorch context
            mat: matrix or batch of matrices of size `(n, n) or (a, n, n)`

        Returns:
            Inverse square root of mat, of shape `(n, n) or (a, n, n)`
        """

        if len(mat.shape) == 2:
            mat = mat.unsqueeze(0)

        norm_mat = torch.norm(mat, dim=[1, 2])
        norm_mat = norm_mat.reshape(mat.size(0), 1, 1)
        eye = (
            torch.eye(
                mat.size(1), requires_grad=False, device=mat.device, dtype=mat.dtype
            )
            .reshape(1, mat.size(1), mat.size(1))
            .repeat(mat.size(0), 1, 1)
        )

        p = mat / norm_mat
        p_sqrt = InvSqrtm.pade_p[0] * eye
        q_sqrt = InvSqrtm.pade_q[0] * eye
        p_app = eye - p
        p_hat = p_app

        for i in range(5):
            p_sqrt += InvSqrtm.pade_p[i + 1] * p_hat
            q_sqrt += InvSqrtm.pade_q[i + 1] * p_hat
            p_hat = p_hat.bmm(p_app)

        mat_sqrt_inv = torch.linalg.solve(p_sqrt, q_sqrt)
        mat_sqrt_inv = mat_sqrt_inv / torch.sqrt(norm_mat)
        ctx.save_for_backward(mat, mat_sqrt_inv, eye)
        return mat_sqrt_inv

    @staticmethod
    def backward(ctx, grad_output):
        mat, mat_sqrt_inv, eye = ctx.saved_tensors
        mat_inv = mat_sqrt_inv.bmm(mat_sqrt_inv)
        grad_lya = -mat_inv.bmm(grad_output).bmm(mat_inv)
        norm_sqrt_inv = torch.norm(mat_sqrt_inv)
        b = mat_sqrt_inv / norm_sqrt_inv
        c = grad_lya / norm_sqrt_inv
        for i in range(8):
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0 * eye - b_2) - b_2.bmm(c) + b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * eye - b_2)
        grad_input = 0.5 * c
        return grad_input


inv_sqrtm = InvSqrtm.apply


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


def check_time_tensor(x: Tensor, arg_name: str, allow_empty=False):
    # check that a time tensor is valid (it must be a 1D tensor sorted in strictly
    # ascending order and containing only positive values)
    if x.ndim != 1:
        raise ValueError(
            f'Argument `{arg_name}` must be a 1D tensor, but is a {x.ndim}D tensor.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument `{arg_name}` must contain at least one element.')
    if not torch.all(torch.diff(x) > 0):
        raise ValueError(
            f'Argument `{arg_name}` must be sorted in strictly ascending order.'
        )
    if not torch.all(x >= 0):
        raise ValueError(f'Argument `{arg_name}` must contain positive values only.')
