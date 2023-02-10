from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from torchqdynamics import hermitian


def mesolve(
    H: torch.Tensor,
    rho0: torch.Tensor,
    times: np.ndarray,
    c_ops: List[torch.Tensor] = None,
    e_ops: List[torch.Tensor] = None,
    n_substeps: int = 1,
    save_states=True,
    progress_bar=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert n_substeps > 0
    c_ops = c_ops or []
    e_ops = e_ops or []

    dt = (times[1] - times[0]) / n_substeps
    c_dag_c = torch.stack([hermitian(op) @ op for op in c_ops]).sum(dim=0)
    R = (
        torch.eye(*H.shape)
        + (-1j * H + 0.5 * c_dag_c) @ (1j * H + 0.5 * c_dag_c) * dt**2
    )
    inv_sqrt_R = _inv_sqrtm(R)

    M_r = torch.eye(*H.shape, dtype=H.dtype) - (1j * H + 0.5 * c_dag_c) * dt
    M_r_tilde = M_r @ inv_sqrt_R

    c_ops_tilde = [op @ inv_sqrt_R for op in c_ops]
    e_ops = torch.stack(e_ops)

    rho = torch.clone(rho0)
    states, measures = [], []

    if progress_bar:
        times = tqdm(times)

    for _ in times:
        for _ in range(n_substeps):
            next_rho = M_r_tilde @ rho
            next_rho = next_rho @ hermitian(M_r_tilde)
            next_rho += sum(
                [
                    c_op_tilde @ rho @ hermitian(c_op_tilde) * dt
                    for c_op_tilde in c_ops_tilde
                ]
            )

            next_rho /= torch.trace(next_rho)
            rho = next_rho

        if len(e_ops) > 0:
            measure = e_ops @ rho
            measure = torch.einsum("bii->b", measure)
            measures.append(measure)

        if save_states:
            states.append(torch.clone(rho))

    states = torch.stack(states) if len(states) > 0 else None
    measures = torch.stack(measures) if len(measures) > 0 else None

    return states, measures


def _inv_sqrtm(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eig(matrix)
    vals = vals.contiguous()
    vals_pow = vals.pow(-0.5)
    matrix_pow = torch.matmul(
        vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs))
    )
    return matrix_pow


def _dissipator(
    L: torch.tensor, Ldag: torch.tensor, rho: torch.tensor
) -> torch.tensor:
    # [speedup] by using the precomputed jump operator adjoint
    return L @ rho @ Ldag - 0.5 * Ldag @ L @ rho - 0.5 * rho @ Ldag @ L


def _lindbladian(
    H: torch.Tensor,
    Ls: torch.Tensor,
    Lsdag: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    # [speedup] by using the precomputed jump operators adjoint
    return -1j * (H @ rho - rho @ H) + _dissipator(Ls, Lsdag, rho).sum(0)
