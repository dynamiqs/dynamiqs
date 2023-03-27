from typing import Tuple, Union

import torch
from torch import Tensor


def kraus_map(rho: Tensor, O: Tensor) -> Tensor:
    """Compute the application of a Kraus map on an input density matrix.

    This is equivalent to `torch.sum(operators @ rho[None,...] @ operators.adjoint(),
    dim=0)`. The use of einsum yields better performances on large matrices, but may
    cause a small overhead on smaller matrices (N <~ 50).

    TODO Fix documentation

    Args:
        rho: Density matrix of shape `(a, ..., n, n)`.
        operators: Kraus operators of shape `(a, b, n, n)`.
    Returns:
        Density matrix of shape `(a, ..., n, n)` with the Kraus map applied.
    """
    return torch.einsum('abij,a...jk,abkl->a...il', O, rho, O.adjoint())


def inv_sqrtm(mat: Tensor) -> Tensor:
    """Compute the inverse square root of a matrix using its eigendecomposition.

    TODO Replace with Schur decomposition once released by PyTorch.
         See the feature request at https://github.com/pytorch/pytorch/issues/78809.
         Alternatively, see a sqrtm implementation at
         https://github.com/pytorch/pytorch/issues/25481#issuecomment-584896176.
    """
    vals, vecs = torch.linalg.eigh(mat)
    return vecs @ torch.linalg.solve(vecs, torch.diag(vals ** (-0.5)), left=False)


def bexpect(operators: Tensor, state: Tensor) -> Tensor:
    """Compute the expectation values of many operators on a quantum state or
    density matrix. The method is batchable over the operators and the state.

    TODO Adapt to both density matrices, kets and bras.

    Args:
        operators: tensor of shape `(b, n, n)`
        state: tensor of shape `(..., n, n)` or `(..., n)`
    Returns:
        expectation value of shape `(..., m)`
    """
    return torch.einsum('bij,...ji->...b', operators, state)


def none_to_zeros_like(
    in_tuple: Tuple[Union[Tensor, None], ...], shaping_tuple: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    """Convert `None` values of `in_tuple` to zero-valued tensors with the same shape
    as `shaping_tuple`."""
    return tuple(
        torch.zeros_like(s) if a is None else a for a, s in zip(in_tuple, shaping_tuple)
    )


def add_tuples(a: Tuple, b: Tuple) -> Tuple:
    """Element-wise sum of two tuples of the same shape."""
    return tuple(x + y for x, y in zip(a, b))
