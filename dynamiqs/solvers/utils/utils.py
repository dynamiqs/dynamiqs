from __future__ import annotations

from functools import partial
from math import sqrt
from typing import Iterator, get_args

import torch
from torch import Tensor
from tqdm import tqdm as std_tqdm

from ..._utils import obj_type_str
from ...time_tensor import TimeTensor, _factory_constant
from ...utils.tensor_types import ArrayLike
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


def iteraxis(x: Tensor, axis: int = 0) -> Iterator[Tensor]:
    for i in range(x.size(axis)):
        yield x.select(axis, i)


def format_L(L: list[Tensor]) -> Tensor:
    # Format a list of tensors of individual shape (n, n) or (?, n, n) into a single
    # batched tensor of shape (nL, bL, n, n). An error is raised if all batched
    # dimensions `?` are not the same.

    n = L[0].size(-1)
    L = [x.view(-1, n, n) for x in L]  # [(?, n, n)] with ? = 1 if not batched

    bL = common_batch_size([x.size(0) for x in L])
    if bL is None:
        L_shapes = [tuple(x.shape) for x in L]
        raise ValueError(
            'Argument `jump_ops` should be a list of 2D arrays or 3D arrays with the'
            ' same batch size, but got a list of arrays with incompatible shapes'
            f' {L_shapes}.'
        )

    L = [x.expand(bL, -1, -1) for x in L]  # [(bL, n, n)]
    return torch.stack(L)


def common_batch_size(dims: list[int]) -> int | None:
    # If `dims` is a list with two values 1 and x, returns x. If it contains only
    # 1, returns 1. Otherwise, returns `None`.
    bs = torch.tensor(dims).unique()
    if (1 not in bs and len(bs) > 1) or len(bs) > 2:
        return None
    return bs.max().item()


def to_time_operator(
    x: ArrayLike | TimeTensor, arg_name: str, dtype: torch.dtype, device: torch.device
) -> TimeTensor:
    if isinstance(x, get_args(ArrayLike)):
        return _factory_constant(x, dtype=dtype, device=device)
    elif isinstance(x, TimeTensor):
        # todo: convert time tensor dtype/device if possible, raise error otherwise
        return x
    else:
        raise TypeError(
            f'Argument `{arg_name}` must be an array-like object or a `TimeTensor`, but'
            f' has type {obj_type_str(x)}.'
        )
