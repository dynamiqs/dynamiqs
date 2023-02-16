from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def mesolve(
    H: torch.Tensor,
    rho0: torch.Tensor,
    t_save: np.ndarray,
    jump_ops: List[torch.Tensor] = None,
    observable_ops: List[torch.Tensor] = None,
    n_substeps: int = 1,
    save_states=True,
    progress_bar=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Master equation solver. The schema is taken from
     https://journals.aps.org/pra/abstract/10.1103/PhysRevA.91.012118 """

    assert n_substeps > 0
    jump_ops = jump_ops or []
    observable_ops = observable_ops or []

    dt = (t_save[1] - t_save[0]) / n_substeps
    c_dag_c = torch.stack([op.adjoint() @ op for op in jump_ops]).sum(dim=0)
    R = (
        torch.eye(*H.shape) +
        (-1j * H + 0.5 * c_dag_c) @ (1j * H + 0.5 * c_dag_c) * dt**2
    )
    inv_sqrt_R = _inv_sqrtm(R)

    M_r = torch.eye(*H.shape, dtype=H.dtype) - (1j * H + 0.5 * c_dag_c) * dt
    M_r_tilde = M_r @ inv_sqrt_R

    jump_ops_tilde = [op @ inv_sqrt_R for op in jump_ops]
    observable_ops = torch.stack(observable_ops) if len(observable_ops) > 0 else []

    rho = torch.clone(rho0)
    states, measures = [], []

    for _ in tqdm(t_save, disable=not progress_bar):
        for _ in range(n_substeps):
            next_rho = M_r_tilde @ rho
            next_rho = next_rho @ M_r_tilde.adjoint()
            next_rho += sum(
                [
                    jump_op_tilde @ rho @ jump_op_tilde.adjoint() * dt
                    for jump_op_tilde in jump_ops_tilde
                ]
            )

            next_rho /= torch.trace(next_rho)
            rho = next_rho

        if len(observable_ops) > 0:
            measure = observable_ops @ rho
            measure = torch.einsum('bii->b', measure)
            measures.append(measure)

        if save_states:
            states.append(torch.clone(rho))

    states = torch.stack(states) if len(states) > 0 else None
    measures = torch.stack(measures) if len(measures) > 0 else None

    return states, measures


def _inv_sqrtm(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    TODO: replace with Schur decomposition method
    See https://www.sciencedirect.com/science/article/pii/002437958380010X

    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
    Returns:
        Square root of a matrix
    """
    vals, vecs = torch.linalg.eig(matrix)
    vals = vals.contiguous()
    vals_pow = vals.pow(-0.5)
    matrix_pow = torch.matmul(
        vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs))
    )
    return matrix_pow


def _dissipator(L: torch.tensor, Ldag: torch.tensor, rho: torch.tensor) -> torch.tensor:
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
