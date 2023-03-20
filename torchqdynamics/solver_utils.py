import torch


def kraus_map(rho, operators):
    """Compute the application of a Kraus map on an input density matrix.

    This is equivalent to `torch.sum(operators @ rho[None,...] @ operators.adjoint(),
    dim=0)`. The use of einsum yields better performances on large matrices, but may
    cause a small overhead on smaller matrices (N <~ 50).

    Args:
        rho: Density matrix of shape (..., n, n).
        operators: Kraus operators of shape (b, n, n).
    Returns:
        Density matrix of shape (..., n, n) with the Kraus map applied.
    """
    return torch.einsum('mij,...jk,mkl->...il', operators, rho, operators.adjoint())


def inv_sqrtm(mat):
    """Compute the inverse square root of a matrix using its eigendecomposition.

    TODO: Replace with Schur decomposition once released by PyTorch.
    See the feature request at https://github.com/pytorch/pytorch/issues/78809.
    Alternatively, see
    https://github.com/pytorch/pytorch/issues/25481#issuecomment-584896176
    for sqrtm implementation.
    """
    vals, vecs = torch.linalg.eigh(mat)
    return vecs @ torch.linalg.solve(vecs, torch.diag(vals**(-0.5)), left=False)
