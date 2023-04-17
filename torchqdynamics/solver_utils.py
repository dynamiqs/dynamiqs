from typing import Tuple, Union

import torch
from torch import Tensor

from .utils import is_ket


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
        return torch.einsum('...ij,bjk,...kl->...b', x.adjoint(), O, x)  # <x|O|x>
    return torch.einsum('bij,...ji->...b', O, x)  # tr(Ox)


def lindbladian(rho: Tensor, H: Tensor, L: Tensor) -> Tensor:
    sum_nojump = (L.adjoint() @ L).sum(dim=0)

    # non-hermitian Hamiltonian
    H_nh = H - 0.5j * sum_nojump

    H_nh_rho = H_nh @ rho
    L_rho_Ldag = kraus_map(rho, L[None, ...])
    return -1j * (H_nh_rho - H_nh_rho.adjoint()) + L_rho_Ldag


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


def hairer_norm(x: Tensor) -> Tensor:
    """Rescaled Frobenius norm of a batched matrix.

    See Equation (4.11) of `Hairer et al., Solving Ordinary Differential Equations I
    (1993), Springer Series in Computational Mathematics`.

    Args:
        x: Tensor of size `(..., n, n)`.

    Returns:
        Tensor of size `(...)` holding the norm of each matrix in the batch.
    """
    return torch.linalg.matrix_norm(x) / torch.sqrt(x.size(-1) * x.size(-2))
